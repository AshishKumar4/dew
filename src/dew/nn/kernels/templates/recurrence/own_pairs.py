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

from dew.nn.kernels import recurrence_algebra as algebra
from dew.nn.kernels.templates import emission, mosaic, native, resources, scan, swizzling
from dew.nn.kernels.templates.cost import Work
from dew.nn.kernels.templates.recurrence.intra import _named, intra_rows, pair_streams, systems
from dew.nn.kernels.templates.recurrence.mainloop import (
    Call,
    Held,
    Operand,
    chunk_liveness,
    f32,
    finished_form,
    fp32_operand,
    held_tile,
    key_block,
    launch_kernel,
    masks,
    stream_blocks,
    stream_tile,
    stream_views,
)


def own_staging(call: Call, keys: int, *, padded: bool = True) -> tuple:
    """The plain tile a key block's keys and its tokens' own decays are staged in for their shifted reads
    (`shifted_keys`): `[2, SUB + C, keys]` fp32, each's block below `SUB` rows a shifted read takes in place
    of the tokens before the chunk (masked); `[2, C, keys]` where not `padded`, for reads that stay in their
    row's sub-chunk (`own_bands`': its pad held an H100's intra_own at three programs a multiprocessor,
    one under its four, M-JS-145k). Its rows are the block's own width apart: the accumulator's
    layout reads a plain tile only at strides whole tiles of 16 columns divide (a stride of 24 does not lower,
    jsf-own-smem). At a 16-column block a warp's rows alternate two runs of banks, two of its four rows a
    wavefront on one; at 32 or 64 columns every row starts on the same bank, its four rows a wavefront's
    conflict, which `key_block`'s choice (by registers and residency) does not weigh."""
    return (swizzling.plain((2, (algebra.SUB if padded else 0) + call.chunk, keys), jnp.float32),)


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
    prefix halves `prefix`, each row against the row before, shifted in registers once a block), each
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


def own_bands(queries, staged, betas, bands, *, products: tuple[str, ...], unit_keys: bool) -> None:
    """Each output's sub-chunks' own pairs of one key block added into `bands`, a plain fp32 shared tile
    `[outputs, C, SUB]` (`algebra.key_diagonal_bands`' form, a row's column `e` its pair `e` tokens back, an
    entry no thread writes zero), from the outputs' rows (`products`': the scores' the queries `queries`
    `[C, keys]`, the system's each key scaled by its write strength `betas` `[C]`, formed as its terms take
    it, `algebra.write_gated`'s one rounding) and the keys and the tokens' own decays `staged`
    (`own_staging`'s, unpadded: a chunk's row `r` at `r`), a thread's own words:
    each quad of the warpgroup's threads a sub-chunk's rows `p` and `SUB - 1 - p` and a quarter of the
    columns each, one loop of `SUB + 1` steps over both rows' pairs at or before them alone (`p + 1` and
    `SUB - p`, so every thread steps alike), its pair's decay the product of its tokens' own formed back from
    its own token as the token recurrence forms it, each term the row times the key, then times the decay
    (or, `unit_keys`, the row times the decay, then times the key), summed over its columns and then the
    quad's (`utils.shfl_bfly`), the warpgroup met once when done. fla_kda's token-parallel diagonal
    (chunk_kda_fwd_intra_token_parallel) walks the pairs at or before a token alone; the bands a tile in the
    accumulator's layout took every offset of every row, its shifted reads whole tiles and its sums over a
    key block's columns a warp's shuffles, 70% of an emulated GH200's intra_own issues (M-JS-145h)."""
    outputs, (chunk, keys) = len(products), staged.shape[1:]
    # Each row's pairs at or before it in its sub-chunk, each over the block's columns: its key and decay
    # read and its decay's multiply (each output's own where `unit_keys`), and each output's row read (the
    # system's its key and write strength, and their multiply), its two multiplies and an add; then each
    # output's sum over the quad (two shuffles and two adds) and its band's entry read, added and written.
    pairs = chunk * (algebra.SUB + 1) // 2
    columns = pairs * keys
    system = "system" in products
    work = Work(operations=columns * (3 * outputs + (outputs if unit_keys else 1) + system)
                + pairs * outputs * 5,
                shared_bytes=4 * columns * (2 + outputs) + 4 * pairs * system + 8 * pairs * outputs)
    refs = tuple(ref for ref in (queries, staged, betas, bands) if ref is not None)
    body = emission.stating(lambda *refs: work)(
        lambda ctx, *refs: _own_bands(*refs, products=products, unit_keys=unit_keys))
    plgpu.inline_mgpu(arg_types=(plgpu.RefType(),) * len(refs), return_type=())(body)(*refs)


def _own_bands(*refs, products: tuple[str, ...], unit_keys: bool) -> None:
    """`own_bands` in Mosaic GPU's terms, a `SPAN` of rows at a time: `refs` its queries (where it forms the
    scores), the staged keys and decays, the write strengths (where it forms the system) and the bands."""
    index, i32, f32_type = ir.IndexType.get(), ir.IntegerType.get_signless(32), ir.F32Type.get()
    scored, system = "scores" in products, "system" in products
    queries_ref = refs[0] if scored else None
    staged_ref, betas_ref, bands_ref = refs[scored], refs[scored + 1] if system else None, refs[-1]
    chunk, keys = ir.MemRefType(staged_ref.type).shape[1:]
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
            return [memref.load(queries_ref, at(row, column)) for column in columns]
        strength = native.Word(memref.load(betas_ref, at(row)))
        keys = (native.Word(memref.load(staged_ref, at(0, arith.addi(row, const(pad)), column)))
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
        _own_span(row_words, staged_ref, bands_ref, short, long, p, columns, leader, one, at, const,
                  products=products, unit_keys=unit_keys, pad=pad)
    utils.warpgroup_barrier()


def _own_span(row_words, staged_ref, bands_ref, short, long, p, columns, leader, one, at, const, *,
              products: tuple[str, ...], unit_keys: bool, pad: int) -> None:
    """`_own_bands`' steps over one span's rows `short` and `long` of each quad (`own_bands`'), each
    output's row's words `row_words(name, row, columns)`, `pad` the staged tile's rows before the chunk's."""
    i32, sub, run, outputs = ir.IntegerType.get_signless(32), algebra.SUB, len(columns), len(products)

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
        return ([memref.load(staged_ref, at(0, source, column)) for column in columns]
                + [memref.load(staged_ref, at(1, own, column)) for column in columns]
                + [word for name in products for word in row_words(name, row, columns)])

    # The steps a loop, carrying the pair's decay (or each output's decayed row, `unit_keys`) and the next
    # step's words, read a step ahead of the terms that take them: unrolled, ptxas held later steps' reads
    # across earlier ones and spilled an H100's 80 registers (jsf-gate-steps3); read in the step, each step's
    # adds waited on its reads, 42% of an emulated GH200's warp time a long scoreboard (M-JS-145k).
    state = run * (outputs if unit_keys else 1)
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
        after, decays = [], []
        if not unit_keys:
            # The pair's decay, one for every output, restarting at a row's own pair.
            decays = [arith.select(first, one, arith.mulf(b, a)) for b, a in zip(before, factor, strict=True)]
            after = decays
        for o in range(outputs):
            values = words[(2 + o) * run:(3 + o) * run]
            if unit_keys:
                decayed = [arith.select(first, value, arith.mulf(b, a))
                           for value, b, a in zip(values, before[o * run:(o + 1) * run], factor, strict=True)]
                after += decayed
                pairs = [None] * run
            else:
                decayed, pairs = values, [native.Word(d) for d in decays]
            # The algebra's own term (`algebra.band_term`), emitted a word at a time.
            terms = [algebra.band_term(native.Word(r), native.Word(k), d, unit_keys=unit_keys).value
                     for r, k, d in zip(decayed, key, pairs, strict=True)]
            total = terms[0]
            for term in terms[1:]:
                total = arith.addf(total, term)
            for distance in (1, 2):
                total = arith.addf(total, utils.shfl_bfly(total, distance))
            with utils.when(leader):
                place = at(o, row, e)
                memref.store(arith.addf(memref.load(bands_ref, place), total), bands_ref, place)
        scf.yield_(after + ahead)


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
                     functools.partial(pair_streams, call, queries=scored), contracted=True)
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
        shifts, *threaded = staging

        def step(index, slots, bands, accumulators):
            slots.acquire_all()

            def operand(name):
                return stream_tile(slots, name, call.given).value().astype(jnp.float32)

            prefix = tuple(stream_tile(slots, name, gates).value() for name in ("highs", "lows"))
            shifted = shifted_keys(call, shifts, operand("keys"), prefix)
            if not by_thread(call):
                found = algebra.key_diagonal_bands(
                    [intra_rows(call, name, operand, beta_ref.at[h, span]) for name in products], shifted,
                    unit_keys=call.unit_keys)
                bands = tuple(band + more for band, more in zip(bands, found, strict=True))
            else:
                # By thread: the scores' queries and the system's write strengths staged after the keys' and
                # decays' (the system's rows the staged keys scaled in the walk), then each thread its own
                # pairs' words (`own_bands`), the bands summed in shared memory.
                queries, betas, summed = _threaded(threaded, products)
                if queries is not None:
                    plgpu.store(queries, operand("queries"), optimized=False)
                if betas is not None:
                    plgpu.store(betas, family.rows_of(beta_ref.at[h, span]), optimized=False)
                mosaic.synced(call.interpret)
                own_bands(queries, shifts, betas, summed, products=products, unit_keys=call.unit_keys)

            @pl.when(index == steps - 1)
            def _():
                for o, (name, ref) in enumerate(zip(products, sums, strict=True)):
                    band = family.tile_of(threaded[-1].at[o]) if threaded else bands[o]
                    _band_form(call, name).store(
                        ref, algebra.band_masked(band, strict=name == "system", scale=scale), row, chunk,
                        h * algebra.SUB, algebra.SUB, family=family, smem=landing[name])

            return bands, accumulators

        def prologue():
            own_zeroed(call, shifts)
            for summed in threaded[-1:]:
                for o in range(len(products)):
                    plgpu.store(summed.at[o], family.as_tile(family.zeros((chunk, algebra.SUB))),
                                optimized=False)

        carry = (() if by_thread(call)
                 else tuple(family.as_tile(family.zeros((chunk, algebra.SUB))) for _ in products))
        return mosaic.Walk(views, step, steps=steps, carry=carry, prologue=prologue)

    blocks = pair_streams(call, keys, queries=scored)
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


def _band_form(call: Call, name: str):
    """The form an output's bands are stored in (`intra_own`'s): the scores' the held operand's, the system's
    `systems`', as the intra walk stores their sums."""
    return dataclasses.replace(call.held, rows=call.rows) if name == "scores" else systems(call)


def _own_staged(call: Call, keys: int, *, products: tuple[str, ...]) -> tuple:
    """`intra_own`'s staging at key block `keys`: its shifted reads' tile (`own_staging`), and where a
    thread forms its own pairs (`by_thread`) the scores' queries, the system's write strengths and the
    outputs' bands summed (`own_bands`'), plain fp32; and each output's bands' landing. The system's rows
    staged too held an A100's intra_own at two programs a multiprocessor (541 us against 481 at three,
    M-JS-160-a100)."""
    outputs, chunk = len(products), call.chunk
    threaded = ()
    if by_thread(call):
        threaded = (*((swizzling.plain((chunk, keys), jnp.float32),) if "scores" in products else ()),
                    *((swizzling.plain((chunk,), jnp.float32),) if "system" in products else ()),
                    swizzling.plain((outputs, chunk, algebra.SUB), jnp.float32))
    staging = (*own_staging(call, keys, padded=not by_thread(call)), *threaded)
    return staging, {name: _band_form(call, name).landing(chunk, algebra.SUB) for name in products}


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
        # own pairs (`_own_span`): its carried decays (each output's decayed row where `unit_keys`), a step's
        # words (a run of the keys, the factors and each output's row) and the next step's, read ahead.
        run = block // QUARTERS
        state = run * (outputs if call.unit_keys else 1)
        words = run * (2 + outputs)
        return Held(phases=([f32(chunk, block)] * 6, [f32(chunk, block)] * 3,
                            [f32(scan.THREADS, state + 2 * words)], [f32(chunk, algebra.SUB)] * 2))
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
                      diagonal=None, added: tuple = ((), (), ()), forms: tuple | None = None):
    """A key block's gradients through a chunk's pairs stored at `at` (its row and column) into `refs`
    (`dq`, `dk`, `dg`): the queries', the keys' through the gated keys (`beta k`) and their own, and the log
    decay's (`q dq + k (beta dk' - dk)`); returns the write gates' terms, each row's `sum(dk' k)`. Each
    gradient has the fp32 tiles of `added`'s refs at its own place added in turn before it is stored in its
    form (`forms`, fp32 each where None) through its landing (`smem`, one a gradient): the other launches'
    parts of a gradient it finishes (`key_backward._key_pairs_back`). Where
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
    for ref, grad, others, form, landing in zip(refs, grads, added, forms, smem, strict=True):
        for other in others:
            (view,) = square.views(other, row, call.chunk, column, keys)
            grad = grad + call.family.tile_of(view)
        form.store(ref, grad, row, call.chunk, column, keys, family=call.family, smem=landing)
    return jnp.sum(d_keyed * key, axis=1)


def key_pairs_own(q, k, hi, lo, beta, d_scores, d_system, taken, factored, memory, call: Call, *,
                  scale: float):
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
    tensor cores' walk two programs a multiprocessor, where this one holds four and issues no products."""
    chunk, width, family = call.chunk, call.width, call.family
    given, gates, square = call.given, call.gates, fp32_operand(call)
    # The block a multiprocessor holds most programs of, at its registers and its shared memory: by its
    # registers alone, an sm86's 99 KB held none of its 32-key block's 106 KB (row 5282). It issues no
    # products, so a product's depth steps it (`key_block`'s `contracted`), as the forward's.
    # The finished queries' and keys' form, one binding the block's fit and the launch read.
    summed = finished_form(call, once=call.group == 1)
    keys = key_block(call, functools.partial(_key_own_held, call),
                     functools.partial(_key_own_staged, call, summed=summed),
                     functools.partial(_keypair_streams, call), contracted=True)
    steps = width // keys

    def program(at, q_ref, k_ref, hi_ref, lo_ref, beta_ref, ds_ref, da_ref, taken_ref, factored_q_ref,
                factored_k_ref, memory_q_ref, memory_k_ref, dq_ref, dk_ref, dg_ref, db_ref, *, streams,
                staging, landing):
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
                call, (dq_ref, dk_ref, dg_ref), (row, h * width + index * keys), keys,
                (landing.get("summed", landing["block"]),) * 2 + (landing["block"],), (query, key, betas),
                (d_query, d_keyed, d_key), diagonal=strip("scores", scale, 0)[:, None],
                added=((factored_q_ref, memory_q_ref), (factored_k_ref, memory_k_ref), ()),
                forms=(summed, summed, square))

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
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=tiles, stages=(1, 2), live=chunk_liveness(7),
                         steps=steps, model=terms,
                         held=_key_own_held(call, keys), staging=staging, landing=landing,
                         name="recurrence_key_pairs_own", issues=False,
                         # The bands read a row at a time, at a traced offset: staged plain.
                         swizzles={f"{name}{p}": swizzling.UNSWIZZLED for name in ("scores", "system")
                                   for p in range(fp32_operand(call).parts)})(
        q, k, hi, lo, beta, *(algebra.own_bands(d, chunk=chunk) for d in (d_scores, d_system)), taken,
        *factored, *memory)


def _keypair_streams(call: Call, keys: int) -> dict:
    """`key_pairs_own`'s streams at key block `keys`: the pairs' gradients' own bands once, an offset a row
    (`algebra.own_bands`), the queries, keys and prefix halves a block at a time."""
    chunk, given, gates, square = call.chunk, call.given, call.gates, fp32_operand(call)
    bands = ((algebra.SUB, chunk), square)
    return stream_blocks(scores=bands, system=bands,
                         queries=((chunk, keys), given), keys=((chunk, keys), given),
                         gates=((chunk, keys), gates), lows=((chunk, keys), gates))


def _key_own_staged(call: Call, keys: int, *, summed: Operand) -> tuple:
    """`key_pairs_own`'s staging at key block `keys`: its shifted reads' tile (`own_staging`; its moved
    gradients shifted in registers, `shifted_up`), and a key block's landings: the fp32 one every gradient
    but the finished queries' and keys' lands in, and theirs in their form `summed` where it is narrower
    (the fp32 one where it is not, each store waiting before the landing's next), what the launch allocates
    being what `key_block` fits (row 11953)."""
    square = fp32_operand(call)
    landing = {"block": square.landing(call.chunk, keys)}
    if summed.stored != square.stored:
        landing["summed"] = summed.landing(call.chunk, keys)
    return own_staging(call, keys), landing


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
