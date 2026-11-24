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
from jax.extend.mlir.dialects import arith, memref, scf

from dew.algebra import scan as prefix_scan
from dew.nn.kernels import recurrence_algebra as algebra
from dew.nn.kernels.templates import emission, mosaic, native, product, resources, scan, swizzling
from dew.nn.kernels.templates.cost import Work
from dew.nn.kernels.templates.recurrence.intra import _named, intra_rows, pair_streams, systems
from dew.nn.kernels.templates.recurrence.mainloop import (
    Call,
    Held,
    Operand,
    chunk_liveness,
    clamped_gradient,
    f32,
    finished_form,
    fp32_operand,
    held_tile,
    key_block,
    launch_kernel,
    limb_blocks,
    limb_tiles,
    limb_views,
    masks,
    stream_blocks,
    stream_tile,
    stream_views,
    widest_layout,
)
from dew.nn.precision import compensated_add


def own_staging(call: Call, keys: int) -> tuple:
    """The plain tile a key block's keys and its tokens' own decays are staged in for their shifted reads
    (`shifted_keys`): `[2, SUB + C, keys]` fp32, each's block below `SUB` rows a shifted read takes in place
    of the tokens before the chunk (masked); a thread's own reads, which stay in their row's sub-chunk, take
    no pad (`_own_rows`: its pad held an H100's intra_own at three programs a multiprocessor, one under its
    four, M-JS-145k). Its rows are the block's own width apart: the accumulator's
    layout reads a plain tile only at strides whole tiles of 16 columns divide (a stride of 24 does not lower,
    jsf-own-smem). At a 16-column block a warp's rows alternate two runs of banks, two of its four rows a
    wavefront on one; at 32 or 64 columns every row starts on the same bank, its four rows a wavefront's
    conflict, which `key_block`'s choice (by registers and residency) does not weigh."""
    return (swizzling.plain((2, algebra.SUB + call.chunk, keys), jnp.float32),)


def _own_rows(call: Call, keyed, decayed, keys: jax.Array, prefix: tuple) -> None:
    """A key block's keys and its tokens' own decays (`algebra.token_decays` of its separated prefix
    `prefix`, each row against the row before, shifted in registers once a block) stored in `keyed` (in its
    dtype, the operands': exact) and `decayed` for `own_bands`' reads, which stay in a row's sub-chunk: the
    last block's reads done first, and the writes seen before this block's (`mosaic.synced`)."""
    family = call.family
    whole = algebra.separated(*prefix)
    earlier = tuple(family.row_shifted(half, 1, period=algebra.SUB, interpret=call.interpret)
                    for half in whole)
    mosaic.synced(call.interpret)
    plgpu.store(keyed, keys.astype(keyed.dtype), optimized=False)
    plgpu.store(decayed, algebra.token_decays(whole, earlier), optimized=False)
    mosaic.synced(call.interpret)


def own_zeroed(call: Call, staged) -> None:
    """`staged`'s (`own_staging`'s) rows below the chunk zeroed, which a shifted read takes for the tokens
    before it and nothing writes: once a program, before its walk, so a row's reads before its sub-chunk are
    finite (`algebra.key_diagonal_pairs_back` masks its strips, not its terms)."""
    pad = staged.shape[1] - call.chunk
    for index in range(2 if pad else 0):
        below = staged.at[index, pl.ds(0, pad)]
        plgpu.store(below, call.family.zeros(below.shape), optimized=False)


def shifted_keys(call: Call, staged, keys: jax.Array, prefix: tuple):
    """A key block's keys and its tokens' own decays (`algebra.token_decays` of its separated prefix, the
    prefix `prefix` (`intra._prefix`'s limbs), each row against the row before, shifted in registers
    once a block), each
    offset's reads of them shifted by it (a sub-chunk a group: the pairs a sub-chunk's own), any value where a
    row's source precedes its sub-chunk (masked): stored once in `staged` (`own_staging`) and read at the
    offset's row. Shifted in registers at every offset (`Family.row_shifted`, a lane's two shuffles and a
    select a register for each), the B200's own pairs ran 350 M warp instructions against fla's whole
    intra's 532 M (M-JS-118)."""
    chunk, family = call.chunk, call.family
    width, pad = keys.shape[1], staged.shape[1] - call.chunk
    whole = algebra.separated(*prefix)
    earlier = tuple(family.row_shifted(half, 1, period=algebra.SUB, interpret=call.interpret)
                    for half in whole)
    # The last block's shifted reads done before its rows are written again, and the writes seen before
    # this block's reads (`mosaic.synced`).
    mosaic.synced(call.interpret)
    for index, value in enumerate((keys, algebra.token_decays(whole, earlier))):
        plgpu.store(staged.at[index, pl.ds(pad, chunk), pl.ds(0, width)], value, optimized=False)
    mosaic.synced(call.interpret)
    return lambda e: tuple(family.tile_of(staged.at[index, pl.ds(pad - e, chunk), pl.ds(0, width)])
                           for index in range(2))


QUARTERS = 4
"""The threads sharing a pair of rows in `own_bands`, each a run of the key block's columns."""

SPAN = 2 * scan.THREADS // QUARTERS
"""The rows `own_bands` takes at once, a pair a quad: a chunk's rows a whole number of them (`by_thread`)."""


def by_thread(call: Call) -> bool:
    """Whether `intra_own` forms its bands a thread's own words (`own_bands`): natively, a chunk a whole
    number of `SPAN`s; else (interpreted, or a narrower chunk) as tiles in the accumulator's layout
    (`algebra.key_diagonal_bands`)."""
    return not call.interpret and call.chunk % SPAN == 0


def own_bands(queries, staged: tuple, betas, bands, *, products: tuple[str, ...], unit_pairs: bool) -> None:
    """Each output's sub-chunks' own pairs of one key block added into `bands`, a plain fp32 shared tile
    `[outputs, C, SUB]` (`algebra.key_diagonal_bands`' form, a row's column its earlier token's place in its
    sub-chunk's diagonal block, an entry no thread writes zero), from the outputs' rows (`products`': the
    scores' the queries `queries` `[C, keys]`, the system's each key scaled by its write strength `betas`
    `[C]`, formed as its terms take it, `algebra.write_gated`'s one rounding) and the keys and the tokens' own
    decays `staged` (`_own_staged`'s two tiles, the keys in the operands' dtype, a chunk's row `r` at `r`), a
    thread's own words: each quad of
    the warpgroup's threads a sub-chunk's rows `p` and `SUB - 1 - p` and a quarter of the columns each, one
    loop of `SUB + 1` steps over both rows' pairs at or before them alone (`p + 1` and `SUB - p`, so every
    thread steps alike), its pair's decay the product of its tokens' own formed back from its own token as the
    token recurrence forms it, each term the row times the key, then times the decay (or, `unit_pairs`, the
    row times the key decayed once for every output, `algebra.decayed_key`), summed over its columns and then
    the quad's (`utils.shfl_bfly`), the warpgroup met once when done. fla_kda's token-parallel diagonal
    (chunk_kda_fwd_intra_token_parallel) walks the pairs at or before a token alone; the bands a tile in the
    accumulator's layout took every offset of every row, its shifted reads whole tiles and its sums over a key
    block's columns a warp's shuffles, 70% of an emulated GH200's intra_own issues (M-JS-145h)."""
    (keyed, decayed), outputs = staged, len(products)
    chunk, keys = keyed.shape
    # Each row's pairs at or before it in its sub-chunk, each over the block's columns: its key and decay
    # read, its decay's multiply and (`unit_pairs`) the decayed key's, and each output's row read (the
    # system's its key and write strength, and their multiply), its multiplies (one where `unit_pairs`, two
    # else) and an add; then each output's sum over the quad (two shuffles and two adds) and its band's
    # entry read, added and written.
    pairs = chunk * (algebra.SUB + 1) // 2
    columns = pairs * keys
    system = "system" in products
    work = Work(operations=columns * ((2 if unit_pairs else 3) * outputs + 1 + unit_pairs + system)
                + pairs * outputs * 5,
                shared_bytes=columns * (keyed.dtype.itemsize * (1 + system) + 4
                                        + (queries.dtype.itemsize if queries is not None else 0))
                + 4 * pairs * system + 8 * pairs * outputs)
    refs = tuple(ref for ref in (queries, keyed, decayed, betas, bands) if ref is not None)
    body = emission.stating(lambda *refs: work)(
        lambda ctx, *refs: _own_bands(*refs, products=products, unit_pairs=unit_pairs))
    plgpu.inline_mgpu(arg_types=(plgpu.RefType(),) * len(refs), return_type=())(body)(*refs)


def _own_bands(*refs, products: tuple[str, ...], unit_pairs: bool) -> None:
    """`own_bands` in Mosaic GPU's terms, a `SPAN` of rows at a time: `refs` its queries (where it forms the
    scores), the staged keys and decays (two tiles), the write strengths (where it forms the system) and the
    bands."""
    index, i32, f32_type = ir.IndexType.get(), ir.IntegerType.get_signless(32), ir.F32Type.get()
    scored, system = "scores" in products, "system" in products
    queries_ref = refs[0] if scored else None
    keyed_ref, decayed_ref = refs[scored], refs[scored + 1]
    betas_ref, bands_ref = refs[scored + 2] if system else None, refs[-1]
    chunk, keys = ir.MemRefType(keyed_ref.type).shape
    pad = 0
    sub, run = algebra.SUB, keys // QUARTERS

    def const(value):
        return arith.constant(i32, value)

    def at(*values):
        def indexed(value):
            return (arith.index_cast(index, value) if isinstance(value, ir.Value)
                    else arith.constant(index, value))
        return [indexed(value) for value in values]

    def row_words(name: str, row, columns) -> list:
        """An output's row's run of words: the queries', or each key times its write strength."""
        if name == "scores":
            return [_widened(memref.load(queries_ref, at(row, column))) for column in columns]
        strength = native.Word(memref.load(betas_ref, at(row)))
        keys = (native.Word(_widened(memref.load(keyed_ref, at(arith.addi(row, const(pad)), column))))
                for column in columns)
        return [algebra.gated(key, strength).value for key in keys]

    thread = arith.remui(utils.thread_idx(), const(scan.THREADS))
    quad, quarter = arith.divui(thread, const(QUARTERS)), arith.remui(thread, const(QUARTERS))
    pair_rows = sub // 2
    p = arith.remui(quad, const(pair_rows))
    columns = [arith.addi(arith.muli(quarter, const(run)), const(m)) for m in range(run)]
    one = arith.constant(f32_type, 1.0)
    leader = arith.cmpi(arith.CmpIPredicate.eq, quarter, const(0))
    for span in range(0, chunk, SPAN):
        first_row = arith.addi(arith.muli(arith.divui(quad, const(pair_rows)), const(sub)), const(span))
        short, long = arith.addi(first_row, p), arith.subi(arith.addi(first_row, const(sub - 1)), p)
        _own_span(row_words, (keyed_ref, decayed_ref), bands_ref, short, long, p, columns, leader, one, at,
                  const, products=products, unit_pairs=unit_pairs, pad=pad)
    utils.warpgroup_barrier()


def _widened(value):
    """A 16-bit word `value` as fp32 (exact), an fp32 one as it is: the staged keys' and queries' words in the
    operands' own dtype (`_own_staged`)."""
    f32_type = ir.F32Type.get()
    return value if value.type == f32_type else arith.extf(f32_type, value)


def _own_span(row_words, staged, bands_ref, short, long, p, columns, leader, one, at, const, *,
              products: tuple[str, ...], unit_pairs: bool, pad: int) -> None:
    """`_own_bands`' steps over one span's rows `short` and `long` of each quad (`own_bands`'), each
    output's row's words `row_words(name, row, columns)`, `staged` the staged keys' and decays' tiles, `pad`
    their rows before the chunk's."""
    i32, sub, run, outputs = ir.IntegerType.get_signless(32), algebra.SUB, len(columns), len(products)
    (keyed_ref, decayed_ref), widened = staged, _widened

    def placed(step):
        """A step's row, its pair's lag `e`, whether it is the row's own pair, and its reads' rows: token
        `i - e`'s keys and token `i - e + 1`'s own decay (any of the chunk's at a row's own pair)."""
        on_short = arith.cmpi(arith.CmpIPredicate.sle, step, p)
        row = arith.select(on_short, short, long)
        e = arith.select(on_short, step, arith.subi(arith.subi(step, const(1)), p))
        first = arith.cmpi(arith.CmpIPredicate.eq, e, const(0))
        source = arith.subi(arith.addi(row, const(pad)), e)
        return row, e, first, source, arith.select(first, source, arith.addi(source, const(1)))

    def read(step) -> list:
        """A step's words: its keys, its factors and each output's row, a run of the quad's columns each."""
        row, _, _, source, own = placed(step)
        return ([widened(memref.load(keyed_ref, at(source, column))) for column in columns]
                + [memref.load(decayed_ref, at(own, column)) for column in columns]
                + [word for name in products for word in row_words(name, row, columns)])

    # The steps a loop, carrying the pair's decay, one for every output, and the next step's words, read a
    # step ahead of the terms that take them: unrolled, ptxas held later steps' reads across earlier ones
    # and spilled an H100's 80 registers (jsf-gate-steps3); read in the step, each step's adds waited on its
    # reads, 42% of an emulated GH200's warp time a long scoreboard (M-JS-145k).
    state = run
    lower, upper, stride = at(0, sub + 1, 1)
    loop = scf.ForOp(lower, upper, stride, [one] * state + read(const(0)))
    with ir.InsertionPoint(loop.body):
        step = arith.index_cast(i32, loop.induction_variable)
        carried = list(loop.inner_iter_args)
        before, words = carried[:state], carried[state:]
        # The next step's words, the last step reading its own again.
        ahead = read(arith.minsi(arith.addi(step, const(1)), const(sub)))
        row, e, first, _, _ = placed(step)
        key, factor = words[:run], words[run:2 * run]
        # The pair's decay, one for every output, restarting at a row's own pair; under `unit_pairs` the key
        # decayed by it once for every output (`algebra.decayed_key`).
        decays = [arith.select(first, one, arith.mulf(b, a)) for b, a in zip(before, factor, strict=True)]
        after = decays
        if unit_pairs:
            key = [algebra.decayed_key(native.Word(k), native.Word(d)).value
                   for k, d in zip(key, decays, strict=True)]
        pairs = [None if unit_pairs else native.Word(d) for d in decays]
        for o in range(outputs):
            values = words[(2 + o) * run:(3 + o) * run]
            # The algebra's own term (`algebra.band_term`), emitted a word at a time.
            terms = [algebra.band_term(native.Word(r), native.Word(k), d, unit_pairs=unit_pairs).value
                     for r, k, d in zip(values, key, pairs, strict=True)]
            total = terms[0]
            for term in terms[1:]:
                total = arith.addf(total, term)
            for distance in (1, 2):
                total = arith.addf(total, utils.shfl_bfly(total, distance))
            with utils.when(leader):
                # The pair's column in its sub-chunk's diagonal block: its earlier token's place there.
                place = at(o, row, arith.subi(arith.remui(row, const(sub)), e))
                memref.store(arith.addf(memref.load(bands_ref, place), total), bands_ref, place)
        scf.yield_(after + ahead)


def shifted_up(call: Call, e, value: jax.Array) -> jax.Array:
    """`value`'s row `i` at row `i - e` (`algebra.key_diagonal_pairs_back`'s `lower`), in registers as
    `Family.row_shifted`, any value where a row's source leaves its sub-chunk (masked)."""
    return call.family.row_shifted(value, -e, period=algebra.SUB, interpret=call.interpret)


def intra_own(q, k, prefix, beta, taken, call: Call, *, scale: float, products: tuple[str, ...],
              into: tuple[list, tuple]) -> None:
    """Each output of the intra walk's (`products`) sub-chunks' own pairs, each its own decay, written in
    place on the sub-chunks' diagonal blocks of the walk's sums `into` (`(refs, groups)`: in-place refs
    `[R, HV * C]` in the outputs' forms, `jax.new_ref`'s, sums of one shape and dtype stacked into one ref, a
    group's, `stacked.groups`), the entries the walk leaves zero (`algebra.key_block_pairs` takes the
    keys before a sub-chunk alone), so each entry is rounded once (`algebra.key_diagonal_bands`' blocks,
    masked as the output takes them, `algebra.block_masked`): a launch of their own, no products on the
    tensor cores, as fla_kda's token-parallel diagonal is (chunk_kda_fwd_intra_token_parallel). It walks each
    chunk's key blocks, the keys, queries and the walk's stored prefix (`intra._prefix`) a block at a time,
    summing each output's blocks and storing them at its last step. In the walk, they held an H100's Kimi
    Linear intra launch at 248 registers, 1982 us against 492 (M-JS-50d). Written as bands `[R, HV * SUB]`
    of pairs by their lag, XLA placed them on the blocks by re-summing both sums whole with the bands
    gathered, 57.5 us of a GH200's Kimi Linear forward of 936 (M-JS-258's base). It issues by the layout whose
    programs a multiprocessor holds most (`mainloop.widest_layout`): no products, its walk waits on its loads,
    beside a producer a GH200 held five programs, 20 consumer warps (its plan, row 5260)."""
    call = dataclasses.replace(call, family=widest_layout(call))
    refs, groups = into
    chunk, width, family = call.chunk, call.width, call.family
    scored = "scores" in products
    keys = key_block(call, functools.partial(_own_held, call, outputs=len(products)),
                     functools.partial(_own_staged, call, products=products),
                     functools.partial(pair_streams, call, queries=scored), contracted=True)
    steps = width // keys

    def program(at, q_ref, k_ref, *refs, streams, staging, landing):
        prefix_refs, (beta_ref, _taken_ref, *held) = refs[:len(prefix)], refs[len(prefix):]
        sums = _summed_views(held, groups)
        n, h = at["chunks"], at["heads"]
        row, span = n * chunk, pl.ds(n * chunk, chunk)
        keyed = h // call.group * width
        views = stream_views(lambda i: (0, i), keys=call.given.views(k_ref, row, chunk, keyed, width),
                             **limb_views(call, prefix_refs, row, chunk, h * width, width),
                             **({"queries": call.given.views(q_ref, row, chunk, keyed, width)}
                                if scored else {}))
        shifts, *threaded = staging if not by_thread(call) else ((), *staging)

        def step(index, slots, bands, accumulators):
            slots.acquire_all()

            def operand(name):
                return stream_tile(slots, name, call.given).value().astype(jnp.float32)

            limbs = tuple(tile.value() for tile in limb_tiles(call, slots))
            if not by_thread(call):
                shifted = shifted_keys(call, shifts, operand("keys"), limbs)
                found = algebra.key_diagonal_bands(
                    [intra_rows(call, name, operand, beta_ref.at[h, span]) for name in products], shifted,
                    unit_pairs=call.unit_pairs)
                bands = tuple(band + more for band, more in zip(bands, found, strict=True))
            else:
                # By thread: the scores' queries and the system's write strengths staged after the keys' and
                # decays' (the system's rows the staged keys scaled in the walk), then each thread its own
                # pairs' words (`own_bands`), the bands summed in shared memory.
                keyed, decayed, *rest = threaded
                queries, betas, summed = _threaded(rest, products)
                _own_rows(call, keyed, decayed, operand("keys"), limbs)
                if queries is not None:
                    plgpu.store(queries, operand("queries").astype(queries.dtype), optimized=False)
                if betas is not None:
                    plgpu.store(betas, family.rows_of(beta_ref.at[h, span]), optimized=False)
                mosaic.synced(call.interpret)
                own_bands(queries, (keyed, decayed), betas, summed, products=products,
                          unit_pairs=call.unit_pairs)

            @pl.when(index == steps - 1)
            def _():
                # Each sub-chunk's diagonal block of each output, in place in the walk's sums (`into`), the
                # entries it leaves zero.
                for o, (name, ref) in enumerate(zip(products, sums, strict=True)):
                    form = _band_form(call, name)
                    if threaded and form.parts == 1:
                        # A one-part form copied from the summed blocks straight, masked, scaled and
                        # rounded as each word is copied: no landing.
                        (view,) = form.views(ref, row, chunk, h * chunk, chunk)
                        _diagonal_copied(threaded[-1].at[o], view, strict=name == "system", scale=scale)
                        continue
                    block = family.tile_of(threaded[-1].at[o]) if threaded else bands[o]
                    masked = algebra.block_masked(block, strict=name == "system", scale=scale)
                    _blocks_written(call, name, masked, ref, row, h * chunk, landing[name])

            return bands, accumulators

        def prologue():
            if shifts:
                own_zeroed(call, shifts)
            for summed in threaded[-1:]:
                for o in range(len(products)):
                    plgpu.store(summed.at[o], family.as_tile(family.zeros((chunk, algebra.SUB))),
                                optimized=False)

        carry = (() if by_thread(call)
                 else tuple(family.as_tile(family.zeros((chunk, algebra.SUB))) for _ in products))
        return mosaic.Walk(views, step, steps=steps, carry=carry, prologue=prologue)

    blocks = pair_streams(call, keys, queries=scored)
    # Each output's pairs a key a term, a sub-chunk's offsets: the model's.
    model = len(products) * 3 * chunk * algebra.SUB * width
    staging, landing = _own_staged(call, keys, products=products)
    inputs = (q, k, *prefix, beta, taken)
    launch_kernel(program, call, out_type=(), grid=(call.chunks, call.heads),
                  grid_names=("chunks", "heads"), blocks=blocks, stages=(1, 2),
                  live=chunk_liveness(len(inputs) - 1), steps=steps, model=model,
                  held=_own_held(call, keys, len(products)),
                  staging=staging, landing=landing, name=_named("recurrence_intra_own", products),
                  issues=False,
                  swizzles={name: resources.widest_swizzle(dtype, keys)
                            for name, (_, dtype) in blocks.items()})(*inputs, *refs)


def _summed_views(refs, groups: tuple[tuple[int, ...], ...]) -> list:
    """Each output's sums as the launch writes them, in the outputs' order: a group of one its ref, a stacked
    group's member its place along the ref's leading axis (`chunk_forward._paired`'s refs)."""
    views: list = [None] * sum(map(len, groups))
    for ref, group in zip(refs, groups, strict=True):
        for place, index in enumerate(group):
            views[index] = ref if len(group) == 1 else ref.at[place]
    return views


def _blocks_written(call: Call, name: str, blocks, ref, row, column, landing) -> None:
    """A chunk's own pairs `blocks` `[C, SUB]` of output `name` (a row's columns its sub-chunk's diagonal
    block's, `algebra.key_diagonal_bands`') written in place on the sub-chunks' diagonal blocks of the walk's
    sums `ref` at `row` and `column`, in the output's form (`_band_form`, each part through `landing`, a
    `[C, SUB]` tile of its stored dtype): a block's rows a tile of 16 no layout of a 64-row product holds,
    so each part is copied from `landing` by its threads' own words (`_diagonal_copied`)."""
    form, chunk, sub = _band_form(call, name), call.chunk, algebra.SUB
    pieces = product.formed(blocks.astype(jnp.float32), form.parts) if form.parts > 1 else (blocks,)
    for piece, view in zip(pieces, form.views(ref, row, chunk, column, chunk), strict=True):
        piece = piece.astype(form.stored)
        if call.interpret:
            for s in range(chunk // sub):
                view.at[pl.ds(s * sub, sub), pl.ds(s * sub, sub)][...] = piece[s * sub:(s + 1) * sub]
            continue
        plgpu.store(landing, piece, optimized=False)
        mosaic.synced(call.interpret)
        _diagonal_copied(landing, view)
        mosaic.synced(call.interpret)


def _diagonal_copied(source, view, *, strict: bool | None = None, scale: float = 1.0) -> None:
    """`source` `[C, SUB]` (a row's columns its sub-chunk's diagonal block's) copied onto the sub-chunks'
    diagonal blocks of `view` `[C, C]`, each thread its run of entries: as it is (a landing of the view's
    dtype), or, given `strict`, an fp32 sum of the blocks taken as `algebra.block_masked` takes it (the pair
    with itself dropped where `strict`, else every word `scale`d) and rounded once to the view's dtype."""
    chunk, sub = source.shape
    entries = chunk * sub
    work = Work(operations=entries * (1 if strict is None else 3),
                shared_bytes=entries * source.dtype.itemsize, device_bytes=entries * view.dtype.itemsize)
    body = emission.stating(lambda *_: work)(
        lambda ctx, from_ref, to_ref: _copied(from_ref, to_ref, strict=strict, scale=scale))
    plgpu.inline_mgpu(arg_types=(plgpu.RefType(), plgpu.RefType()), return_type=())(body)(source, view)


def _copied(source_ref, view_ref, *, strict: bool | None, scale: float) -> None:
    """`_diagonal_copied` in Mosaic GPU's terms: a thread's `C * SUB / THREADS` entries, each its row and
    column in `source` and its sub-chunk's block's column in `view`."""
    index, i32 = ir.IndexType.get(), ir.IntegerType.get_signless(32)
    chunk, sub = ir.MemRefType(source_ref.type).shape
    stored = ir.MemRefType(view_ref.type).element_type
    run = chunk * sub // scan.THREADS

    def const(value):
        return arith.constant(i32, value)

    def at(*values):
        return [arith.index_cast(index, value) for value in values]

    def taken(value, r, c):
        """A sum as its output takes it (`algebra.block_masked`), rounded to the view's dtype once."""
        if strict:
            kept = arith.cmpi(arith.CmpIPredicate.ult, c, arith.remui(r, const(sub)))
            value = arith.select(kept, value, arith.constant(ir.F32Type.get(), 0.0))
        else:
            value = arith.mulf(value, arith.constant(ir.F32Type.get(), scale))
        return value if stored == value.type else arith.truncf(stored, value)

    thread = arith.remui(utils.thread_idx(), const(scan.THREADS))
    for step in range(run):
        entry = arith.addi(arith.muli(thread, const(run)), const(step))
        r, c = arith.divui(entry, const(sub)), arith.remui(entry, const(sub))
        column = arith.addi(arith.muli(arith.divui(r, const(sub)), const(sub)), c)
        value = memref.load(source_ref, at(r, c))
        memref.store(value if strict is None else taken(value, r, c), view_ref, at(r, column))


def _band_form(call: Call, name: str):
    """The form an output's bands are stored in (`intra_own`'s): the scores' the held operand's, the system's
    `systems`', as the intra walk stores their sums."""
    return dataclasses.replace(call.held, rows=call.rows) if name == "scores" else systems(call)


def _own_staged(call: Call, keys: int, *, products: tuple[str, ...]) -> tuple:
    """`intra_own`'s staging at key block `keys`: its shifted reads' tile (`own_staging`); or where a thread
    forms its own pairs (`by_thread`) the keys and the tokens' own decays (`_own_rows`), the scores' queries,
    the system's write strengths and the outputs' blocks summed (`own_bands`'), plain, the keys and queries in
    the operands' dtype, the rest fp32; and each output's blocks' landing. The system's rows
    staged too held an A100's intra_own at two programs a multiprocessor (541 us against 481 at three,
    M-JS-160-a100)."""
    outputs, chunk, given = len(products), call.chunk, call.given.dtype
    if not by_thread(call):
        staging = own_staging(call, keys)
    else:
        # The keys and queries in the operands' own dtype, exact, the tokens' own decays fp32: staged fp32,
        # they held an H100's Kimi Linear own pairs at five programs a multiprocessor (row 5260).
        staging = (swizzling.plain((chunk, keys), given), swizzling.plain((chunk, keys), jnp.float32),
                   *((swizzling.plain((chunk, keys), given),) if "scores" in products else ()),
                   *((swizzling.plain((chunk,), jnp.float32),) if "system" in products else ()),
                   swizzling.plain((outputs, chunk, algebra.SUB), jnp.float32))
    # A landing where a form of parts takes its blocks through registers (`_blocks_written`); a one-part
    # form's blocks copied from their sums straight (`_diagonal_copied`).
    landed = [name for name in products if not by_thread(call) or _band_form(call, name).parts > 1]
    return staging, {name: _band_form(call, name).landing(chunk, algebra.SUB) for name in landed}


def _threaded(threaded: list, products: tuple[str, ...]) -> tuple:
    """`_own_staged`'s by-thread scratch `(queries, betas, bands)`, None where not taken."""
    found = iter(threaded)
    queries = next(found) if "scores" in products else None
    betas = next(found) if "system" in products else None
    return queries, betas, next(found)


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
    if by_thread(call):
        # The bands summed in shared memory (`own_bands`): the keys' and decays' forming, then an
        # output's rows formed (its operand, its write strengths and the rows), then a thread's loop over its
        # own pairs (`_own_span`): its carried pairs' decays, a step's words (a run of the keys, the factors
        # and each output's row) and the next step's, read ahead.
        run = block // QUARTERS
        state = run
        words = run * (2 + outputs)
        return Held(phases=([f32(chunk, block)] * 6, [f32(chunk, block)] * 3,
                            [f32(scan.THREADS, state + 2 * words)], [f32(chunk, algebra.SUB)] * 2))
    return Held(carried=[f32(chunk, algebra.SUB)] * outputs, phases=(
        [f32(chunk, block)] * 6,
        [f32(chunk, block)] * (outputs + 6),
        [f32(chunk, algebra.SUB)] * 2))


def pairs_back_views(call: Call, row: int, h, q_ref, k_ref, prefix_refs: tuple) -> dict:
    """The key-block streams of a key pairs backward's program at chunk `row` and value head `h`
    (`key_pairs_own`'s, `key_backward._key_pairs_back`'s): the queries, keys and the prefix (`intra._prefix`)
    a key block a step."""
    chunk, width, given = call.chunk, call.width, call.given
    keyed = h // call.group * width
    return stream_views(lambda i: (0, i),
                        queries=given.views(q_ref, row, chunk, keyed, width),
                        keys=given.views(k_ref, row, chunk, keyed, width),
                        **limb_views(call, prefix_refs, row, chunk, h * width, width))


def pairs_back_stored(call: Call, refs: tuple, at: tuple, keys: int, smem, operands: tuple, grads: tuple,
                      diagonal=None, added: tuple = ((), (), ()), forms: tuple | None = None,
                      finished: tuple = (None, None, None)):
    """A key block's gradients through a chunk's pairs stored at `at` (its row and column) into `refs`
    (`dq`, `dk`, `dg`): the queries', the keys' through the gated keys (`beta k`) and their own, and the log
    decay's (`q dq + k (beta dk' - dk)`); returns the write gates' terms, each row's `sum(dk' k)`. Each
    gradient has the fp32 tiles of `added`'s refs at its own place added in turn before it is stored in its
    form (`forms`, fp32 each where None) through its landing (`smem`, one a gradient): the other launches'
    parts of a gradient it finishes (`key_backward._key_pairs_back`), then `finished`'s function of it where
    one is given (`key_pairs_own`'s decay gradient's suffix sums). Where
    `diagonal` gives a token's own scores pair's gradient `[C, 1]` (scaled), `grads` were summed without
    it and it joins the queries' and keys' gradients alone: the log decay's terms never take it, a token's
    write read by its own query undecayed, its two halves `q (dS k)` and `k (dS q)` equal, each the size of
    `q k dS` where their difference is none, which summed in left their rounding on every earlier token's
    suffix sum, 4e-10 at Kimi Linear's last token where float64's was 2e-17 (row 7906)."""
    (query, key, betas), (d_query, d_keyed, d_key), (row, column) = operands, grads, at
    own_query, own_key = (0.0, 0.0) if diagonal is None else (diagonal * key, diagonal * query)
    square = fp32_operand(call)
    forms = (square,) * 3 if forms is None else forms
    grads = (d_query + own_query, d_keyed * betas[:, None] + d_key + own_key,
             query * d_query + key * (d_keyed * betas[:, None] - d_key))
    for ref, grad, others, form, landing, finish in zip(refs, grads, added, forms, smem, finished,
                                                         strict=True):
        for other in others:
            (view,) = square.views(other, row, call.chunk, column, keys)
            grad = grad + call.family.tile_of(view)
        grad = grad if finish is None else finish(grad)
        form.store(ref, grad, row, call.chunk, column, keys, family=call.family, smem=landing)
    return jnp.sum(d_keyed * key, axis=1)


def key_pairs_own(q, k, prefix, beta, d_scores, d_system, taken, factored, memory, decays: tuple, ending,
                  call: Call, *, scale: float, clamp=None):
    """`(dq, dk, dg [R, HV * Dk], dbeta [HV, R])`: each chunk and value head's gradients through its
    sub-chunks' own pairs, each its own decay (`algebra.key_diagonal_pairs_back`), given the pairs'
    gradients `d_scores` and `d_system` `[R, HV * C]`: a launch of their own, no products on the tensor
    cores, as the forward's (`intra_own`), each key block's offsets reading the pairs' gradients' own
    bands (`algebra.own_bands`, taken once from the squares, laid an offset a row `[HV * SUB, R]`: each
    offset's strip a row read as a row's vector) and the keys and the tokens' own decays shifted
    (`shifted_keys`). In `_key_pairs_back`'s walk, they held an H100's Kimi Linear launch at 248
    registers, 2268 us against 672 (M-JS-50d). The queries' and keys' it finishes: the factored pairs'
    `factored` (`key_backward._key_pairs_back`'s `(dq, dk)`) and then the memory's `memory`
    (`_key_memory_back`'s) added as it stores them, in `finished_form`'s form (stored once where a key head
    has one value head, nothing summed after); the log decay's and beta's
    fp32. Added in XLA, the three fp32 parts were read again and rounded apart, 343 us of Kimi Linear's
    training on an A100 (M-JS-217); added in the factored pairs' launch, its reads cost it 187 of them, a
    tensor cores' walk two programs a multiprocessor, where this one holds four and issues no products.
    The log decay's it finishes too, summed back over its chunk: the factored pairs' and the memory's terms
    (`decays`, `[R, HV * Dk]` fp32 each) added to its own, then each column's suffix sums from the chunk's
    last token down, each started at its memory decay's term `ending` `[chunks, HV * Dk]` (the last token's,
    `sum(h dh) exp2(g_last)`), compensated (`scan.suffix_columns`; interpreted, `_suffix_summed`'s).
    In XLA, the terms' sum and the last token's term were a pass over them and the suffix sums a batched
    fp32 product, 197 and 209 us of Kimi Linear's training on an A100 (M-JS-222). A clamped call's
    (`Call.clamped`) its clamp's derivative applied as it is stored, from the call's log decays laid
    `clamp` `[R, HV * Dk]` (`mainloop.clamped_gradient`): in XLA, a pass over it, 149 us (M-JS-226)."""
    chunk, width, family = call.chunk, call.width, call.family
    given, square = call.given, fp32_operand(call)
    # The block a multiprocessor holds most programs of, at its registers and its shared memory: by its
    # registers alone, an sm86's 99 KB held none of its 32-key block's 106 KB (row 5282). It issues no
    # products, so a product's depth steps it (`key_block`'s `contracted`), as the forward's.
    # The finished queries' and keys' form, one binding the block's fit and the launch read.
    summed = finished_form(call, once=call.group == 1)
    keys = key_block(call, functools.partial(_key_own_held, call),
                     functools.partial(_key_own_staged, call, summed=summed),
                     functools.partial(_keypair_streams, call), contracted=True)
    steps = width // keys

    def program(at, q_ref, k_ref, *refs, streams, staging, landing):
        prefix_refs, refs = refs[:len(prefix)], refs[len(prefix):]
        (beta_ref, ds_ref, da_ref, _taken_ref, factored_q_ref, factored_k_ref, memory_q_ref, memory_k_ref,
         factored_g_ref, memory_g_ref, ending_ref, *clamp_ref, dq_ref, dk_ref, dg_ref, db_ref) = refs
        shifts, *suffix = staging
        n, h = at["chunks"], at["heads"]
        row, span = n * chunk, pl.ds(n * chunk, chunk)
        # The pairs' own bands' gradients once, the same tile every step, an offset a row
        # (`algebra.own_bands`).
        views = {**stream_views(lambda i: (0, 0),
                                scores=square.views(ds_ref, h * algebra.SUB, algebra.SUB, row, chunk),
                                system=square.views(da_ref, h * algebra.SUB, algebra.SUB, row, chunk)),
                 **pairs_back_views(call, row, h, q_ref, k_ref, prefix_refs)}

        def step(index, slots, d_betas, accumulators):
            slots.acquire_all()
            betas = family.rows_of(beta_ref.at[h, span])
            query = stream_tile(slots, "queries", given).value().astype(jnp.float32)
            key = stream_tile(slots, "keys", given).value().astype(jnp.float32)
            limbs = tuple(tile.value() for tile in limb_tiles(call, slots))
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
                shifted_keys(call, shifts, key, limbs), functools.partial(shifted_up, call),
                (zero, zero), zero)
            column = h * width + index * keys
            start = family.columns_of(ending_ref.at[n, pl.ds(column, keys)])
            # The clamp's log decays' view, read once the sums are formed (`_suffix_summed`): read here, its
            # tile lived through the queries' and keys' stores, an sm100's launch spilling 8 bytes a thread.
            floor = square.views(clamp_ref[0], row, chunk, column, keys)[0] if clamp_ref else None
            d_betas = d_betas + pairs_back_stored(
                call, (dq_ref, dk_ref, dg_ref), (row, column), keys,
                (landing.get("summed", landing["block"]),) * 2 + (landing["block"],), (query, key, betas),
                (d_query, d_keyed, d_key), diagonal=strip("scores", scale, 0)[:, None],
                added=((factored_q_ref, memory_q_ref), (factored_k_ref, memory_k_ref),
                       (factored_g_ref, memory_g_ref)),
                forms=(summed, summed, square),
                finished=(None, None, functools.partial(_suffix_summed, call, suffix, start, floor)))

            @pl.when(index == steps - 1)
            def _():
                db_ref.at[h, span][...] = d_betas

            return d_betas, accumulators

        return mosaic.Walk(views, step, steps=steps, carry=family.as_row(family.zeros((chunk,))),
                           prologue=lambda: own_zeroed(call, shifts))

    tiles = _keypair_streams(call, keys)
    staging, landing = _key_own_staged(call, keys, summed=summed)
    fp32_rows, summed_rows = square.shape(call.heads * width), summed.shape(call.heads * width)
    betas = jax.ShapeDtypeStruct((call.heads, call.rows), jnp.float32)
    out_type = (summed_rows, summed_rows, fp32_rows, betas)
    # Each offset's two pairs' terms a key, forward and moved up: the model's.
    terms = 2 * 2 * 3 * chunk * algebra.SUB * width
    inputs = (q, k, *prefix, beta, *(algebra.own_bands(d, chunk=chunk) for d in (d_scores, d_system)), taken)
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=tiles, stages=(1, 2),
                         live=chunk_liveness(len(inputs) - 1), steps=steps, model=terms,
                         held=_key_own_held(call, keys), staging=staging, landing=landing,
                         name="recurrence_key_pairs_own", issues=False,
                         # The bands read a row at a time, at a traced offset: staged plain.
                         swizzles={f"{name}{p}": swizzling.UNSWIZZLED for name in ("scores", "system")
                                   for p in range(fp32_operand(call).parts)})(
        *inputs, *factored, *memory, *decays, ending, *(() if clamp is None else (clamp,)))


def _keypair_streams(call: Call, keys: int) -> dict:
    """`key_pairs_own`'s streams at key block `keys`: the pairs' gradients' own bands once, an offset a row
    (`algebra.own_bands`), the queries, keys and prefix a block at a time."""
    chunk, given, square = call.chunk, call.given, fp32_operand(call)
    bands = ((algebra.SUB, chunk), square)
    return stream_blocks(scores=bands, system=bands,
                         queries=((chunk, keys), given), keys=((chunk, keys), given),
                         **limb_blocks(call, chunk, keys))


def _key_own_staged(call: Call, keys: int, *, summed: Operand) -> tuple:
    """`key_pairs_own`'s staging at key block `keys`: its shifted reads' tile (`own_staging`; its moved
    gradients shifted in registers, `shifted_up`), and a key block's landings: the fp32 one every gradient
    but the finished queries' and keys' lands in, and theirs in their form `summed` where it is narrower
    (the fp32 one where it is not, each store waiting before the landing's next), what the launch allocates
    being what `key_block` fits (row 11953); natively, the decay gradient's suffix sums' tile."""
    square = fp32_operand(call)
    landing = {"block": square.landing(call.chunk, keys)}
    if summed.stored != square.stored:
        landing["summed"] = summed.landing(call.chunk, keys)
    # The decay gradient's suffix sums' tile natively, its terms' rows and each column's start
    # (`scan.suffix_columns`).
    suffix = () if call.interpret else (swizzling.plain((call.chunk + 1, keys), jnp.float32),)
    return (*own_staging(call, keys), *suffix), landing


def _suffix_summed(call: Call, suffix: list, start, floor, terms: jax.Array) -> jax.Array:
    """A key block's decay gradient `terms` `[C, keys]` summed back over its chunk, each column started at
    `start` (`key_pairs_own`'s), and where `floor` views its log decays `[C, keys]` (a clamped call's) the
    clamp's derivative applied (`clamped_gradient`), read once the sums are: natively through the staged tile
    `suffix` (its terms' rows, then the starts' row, `scan.suffix_columns`), read back in the terms' layout
    (`Family.tile_of`); interpreted, as a tile: the compensated suffix sums (`dew.algebra.scan.compensated`),
    each column's start added to its pair, each sum rounded once."""
    if call.interpret:
        terms = terms.astype(jnp.float32)
        high, low = prefix_scan.compensated(terms, axis=0, reverse=True)
        started = (jnp.asarray(start, jnp.float32)[None], jnp.zeros_like(high[:1]))
        summed = compensated_add((high, low), started)[0]
    else:
        (tile,), chunk = suffix, call.chunk
        plgpu.store(tile.at[pl.ds(0, chunk)], terms, optimized=False)
        tile[chunk, :] = start
        mosaic.synced(call.interpret)
        scan.suffix_columns(tile)
        summed = call.family.tile_of(tile.at[pl.ds(0, chunk)])
    return summed if floor is None else clamped_gradient(call.family.tile_of(floor), summed)


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
