"""The recurrence template's intra launches under a decay a key dimension (Kimi delta attention's): each
chunk's pairs within it, the scores and the keys' system, walked a key block at a time on the mainloop's
skeleton (`mainloop.launch_kernel`), their factors formed once for both (the algebra's `key_block_pairs`),
and the system's inverse by doubling (`_intra_solve`), as fla_kda's intra rows and solve; each sub-chunk's
own pairs are `own_pairs`'. A seam of the mainloop of its own: these launches read its primitives, and
nothing of the mainloop reads them."""

import dataclasses
import functools
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation, recurrence_algebra as algebra
from dew.nn.kernels.templates import bundles, mosaic, operands, product, resources, scan, swizzling
from dew.nn.kernels.templates.operands import Issuer
from dew.nn.kernels.templates.recurrence.mainloop import (
    Call,
    Held,
    Operand,
    _doubling_held,
    _inverse_flops,
    _merges,
    _merging,
    building,
    bundle_dtype,
    chunk_liveness,
    f32,
    held_tile,
    key_block,
    launch_kernel,
    lowered_once,
    masks,
    product_fragments,
    shared_allocation,
    solve_persistent,
    staging_tiles,
    stream_blocks,
    stream_tile,
    stream_views,
)
from dew.nn.precision import compensated_add


def systems(call: Call) -> Operand:
    """The form `_intra` passes each chunk's keys' system to `_intra_solve` in: fp32, as fla's intra rows
    store their `Akk` for its solve."""
    return Operand(call.rows, 1, jnp.dtype(jnp.float32))


@lowered_once
def _intra_solve(system, taken, call: Call):
    """`(I + A)^-1` less its unit diagonal `[R, HV * C]` as `call.held` of each chunk and value head's keys'
    system `A` (`_intra`'s, `systems`' form), inverted by doubling on the chunk's tile (`algebra.doubled`):
    fla_kda's solve. Fused with the pairs, the doubling's tiles beside theirs held an H100's thread at 168
    registers and one program a multiprocessor, 12.5% of its warps' cycles issuing, the launch 0.944 ms of
    Kimi Linear's 1.86 ms forward (kda-ncu-h100-1170e4d25)."""
    chunk = call.chunk
    held, solved, family = dataclasses.replace(call.held, rows=call.rows), systems(call), call.family

    def program(at, a_ref, taken_ref, out_ref, *, streams, staging, landing):
        n, h = at["chunks"], at["heads"]
        row = n * chunk
        views = stream_views(system=solved.views(a_ref, row, chunk, h * chunk, chunk))

        def step(index, slots, carry, accumulators):
            del index
            lower = algebra.solved(stream_tile(slots, "system", solved).value(),
                                   _merging(tuple(staging), call))
            held.store(out_ref, lower, row, chunk, h * chunk, chunk, family=family, smem=landing["square"])
            return carry, accumulators

        return mosaic.Walk(views, step)

    return launch_kernel(program, call, out_type=held.shape(call.heads * chunk),
                         grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"),
                         blocks=stream_blocks(system=((chunk, chunk), solved)),
                         stages=2 if solve_persistent(call) else 1,
                         live=chunk_liveness(1), steps=1, model=_inverse_flops(chunk),
                         held=Held(phases=(_doubling_held(call),)),
                         staging=_merges(call), landing={"square": held.landing(chunk, chunk)},
                         persistent=solve_persistent(call), reduces=False)(
        system, taken)


@building(lambda launch: ("recurrence_intra", "recurrence_intra_scores", "recurrence_intra_system",
                          "recurrence_prefix_halves"))
def _intra(q, k, g, beta, taken, call: Call, *, scale: float, solves: bool):
    """`(Aqk [R, HV * C], A [R, HV * C] where the call `solves`, hi, lo [R, HV * Dk])` of each chunk and value
    head under a decay a key dimension: the decay's prefix halves (`call.gates`, `algebra.prefix`'s pair)
    from the laid decay `g` `[R, HV * Dk]` fp32, which every later launch reads (`_prefix_halves`), then the
    scores as `call.held` and the keys' system where it `solves` (`systems`' form, the write-gated keys'
    pairs, strict), which `_intra_solve` inverts, in one walk sharing every factor (`_sub_chunk_walk`): each
    pair of sub-chunks' decay factored (`algebra.key_block_pairs`), their products on the tensor cores
    (fla_kda's intra rows), each sub-chunk's own pairs zero (a launch of their own, `own_pairs.intra_own`)."""
    products = ("scores", "system") if solves else ("scores",)
    # The decay's prefix halves a launch of their own on the call's bundle (fla_kda's chunk_local_cumsum), the
    # walk of the pairs reading them on the sub-chunk's (`_sub_chunk_walk`).
    hi, lo = _prefix_halves(g, taken, call)
    return (*_sub_chunk_walk(q, k, hi, lo, beta, taken, sub_chunked(call), scale=scale, products=products),
            hi, lo)


def sub_chunked(call: Call) -> Call:
    """`call` on the bundle whose products take a sub-chunk's rows (`bundles.layouts` at `algebra.SUB`
    rows: mma.sync's on every target, an H100's and a B200's among them, as a decode step's 8 rows take it),
    the intra walk's: a sub-chunk's row of pairs a product, fla_kda's inter-sub-chunk blocks
    (chunk_kda_fwd_kernel_inter_solve_fused). On warpgroup MMA's 64 rows each sub-chunk's product took a
    whole chunk's rows masked to its own, 2.3 times fla's inter-sub-chunk launch on a GH200 and 13 times
    Dew's port of it on a B200 (M-JS-144-gh200: 330 + 87 us against 142 with the solve; js106c: 712 + 664
    against 102)."""
    family = pairs_family(call.target.generation, call.given.dtype) or call.family
    if family.rows <= algebra.SUB:
        # Its warps along a sub-chunk's row of pairs' columns, its sums' layout `[SUB, C]`'s (`specialized`).
        family = family.specialized(algebra.SUB, call.chunk)
    return call if family == call.family else dataclasses.replace(call, family=family)


def pairs_family(generation_name: str, dtype) -> mosaic.Family | None:
    """The bundle the intra walk's pairs issue by on `generation_name` (`sub_chunked`): the one whose products
    take a sub-chunk's rows, unspecialized; None where no bundle claims the target."""
    found = bundles.layouts(generation_name, bundle_dtype(dtype), algebra.SUB, scoped=True)
    return found[0] if found else None


def _later(call: Call) -> int:
    """The sub-chunks past a chunk's first, each a row of pairs (`algebra.key_block_pairs`' products)."""
    return call.chunk // algebra.SUB - 1


def _named(launch: str, products: tuple[str, ...]) -> str:
    """The name of `launch`'s walk of `products`: its own where it walks every output (the scores and the
    system), else `<launch>_<output>`, so a walk of one output is a kernel of its own name beside the walk of
    both, which a training call's backward and forward each launch (a kernel's name is its identity,
    `report.named`: Kimi's sm90 training call ran two `recurrence_intra` bodies under one name)."""
    return launch if len(products) > 1 else "_".join((launch, *products))


def _scan_zeroed(call: Call, scanned, keys: int) -> None:
    """`scanned`'s `C` zero rows of each half, below the rows the scan writes (`_intra_prefix`), which its
    shifts read and nothing writes again: once a program, before its walk. Stored every key block, they took
    7% of a B200's Kimi Linear intra walk's cycles (an emulated B200's line map at 3fd15d5978b)."""
    for half in (0, 1):
        plgpu.store(scanned.at[half, pl.ds(0, call.chunk)], call.family.zeros((call.chunk, keys)),
                    optimized=False)


def _intra_prefix(call: Call, scanned, decays: jax.Array, keys: int) -> tuple:
    """A key block's prefix halves (`algebra.scanned_prefix`) of its `decays` `[C, keys]`, staged in `scanned`
    (each half's `C` zero rows, `_scan_zeroed`'s, then its running pair, which the shifts read), left there at
    rows `C` on and returned in the accumulator's layout. The scan's steps run in a row-major layout, each
    thread's columns consecutive, whose transfers at any row offset touch every bank once: in the
    accumulator's, a fragment's 8 rows `keys` words apart shared 8 banks (19.0M bank conflicts on loads in a
    GH200's Kimi Linear intra launch, M-JS-17); the decay laid in and the prefix out once each."""
    chunk = call.chunk
    def half_at(half, rows):
        return scanned.at[half, rows]

    def stage(pair):
        _staged_rows(call, scanned, pair)
        return lambda shift: tuple(_rows_of(half_at(half, pl.ds(chunk - shift, chunk)), keys)
                                   for half in (0, 1))

    mosaic.synced(call.interpret)
    plgpu.store(half_at(0, pl.ds(chunk, chunk)), decays, optimized=False)
    mosaic.synced(call.interpret)
    stage(algebra.scanned_prefix(_rows_of(half_at(0, pl.ds(chunk, chunk)), keys), stage, chunk=chunk))
    return tuple(call.family.tile_of(half_at(half, pl.ds(chunk, chunk))) for half in (0, 1))


def _staged_rows(call: Call, scanned, pair: tuple) -> None:
    """`pair`'s two `[C, keys]` tiles stored over the scan's rows (`scanned`'s rows `C` on,
    `_intra_prefix`'s): a thread reads other threads' rows, so their last reads are done before the rows are
    written again, and the writes made visible before they are read (`mosaic.synced`)."""
    chunk = call.chunk
    mosaic.synced(call.interpret)
    for half, value in enumerate(pair):
        plgpu.store(scanned.at[half, pl.ds(chunk, chunk)], value, optimized=False)
    mosaic.synced(call.interpret)


def _rows_of(ref, keys: int) -> jax.Array:
    """A `[C, keys]` tile `ref` of plain shared rows read in a row-major layout (`plgpu.Layout.WG_STRIDED`):
    each thread its consecutive words, a warp's transfer a contiguous run."""
    rows = ref.shape[0]
    layout = plgpu.Layout.WG_STRIDED((rows, keys), vec_size=min(4, rows * keys // 128))
    return plgpu.load(ref, layout=layout, optimized=False)


def intra_block_stored(call: Call, name: str, block, ref, landing: dict, *, row, head) -> None:
    """The intra walk's sub-chunk row block `[SUB, C]` of output `name` at `row`, a sub-chunk's rows of the
    chunk's square."""
    chunk = call.chunk
    form, smem = ((dataclasses.replace(call.held, rows=call.rows), landing["square"]) if name == "scores"
                  else (systems(call), landing["system"]))
    form.store(ref, block, row, algebra.SUB, head * chunk, chunk, family=call.family, smem=smem)


def _sum_index(call: Call, output: int, later: int) -> int:
    """`_sub_chunk_walk`'s sum of output `output`'s and the `later`-th sub-chunk past the first's."""
    return output * _later(call) + later


@lowered_once
def _prefix_halves(g, taken, call: Call):
    """`(hi, lo) [R, HV * Dk]` fp32: the laid decay `g`'s prefix halves within each chunk (`algebra.prefix`'s
    pair, `call.gates`), a program a chunk, value head and key block (`_running`), each its block's one
    load, scan and store: a running compensated sum down each column where the block fills the warpgroup's
    threads (`_running_prefix`), else the doubling scan (`_intra_prefix`). fla_kda's chunk_local_cumsum, a
    launch the sub-chunk walk and every later launch read (`_sub_chunk_walk`)."""
    chunk, width, gates, family = call.chunk, call.width, call.gates, call.family
    running = _running(call)
    keys = scan.THREADS if running else math.gcd(width, 2 * algebra.SUB)

    def program(at, g_ref, taken_ref, hi_ref, lo_ref, *, streams, staging, landing):
        del taken_ref
        n, h, b = at["chunks"], at["heads"], at["blocks"]
        row, column = n * chunk, h * width + b * keys
        views = stream_views(decays=gates.views(g_ref, row, chunk, column, keys))
        (rows,) = staging

        def step(index, slots, carry, accumulators):
            del index
            slots.acquire_all()
            tile = stream_tile(slots, "decays", gates)
            decays = tile.value()
            found = None
            if running:
                _running_prefix(call, rows, decays, keys)
            else:
                _scan_zeroed(call, rows, keys)
                found = _intra_prefix(call, rows, decays, keys)
            # A running sum's halves through its decays' slot in turn where a TMA store lands them, read and
            # staged before (each store waited; the operand's form and tile, a landing's own), else through
            # one landing in turn; the doubling's through their own.
            slot = tile.refs[0] if not family.stores_through_registers else landing["prefix0"]
            for i, ref in enumerate((hi_ref, lo_ref)):
                # A running sum's halves read as each is stored, one tile held at a time: both at once
                # spilled the program's 128 registers at two programs a multiprocessor (M-JS-145g).
                half = family.tile_of(rows.at[i]) if found is None else found[i]
                gates.store(ref, half, row, chunk, column, keys, family=family,
                            smem=slot if running else landing[f"prefix{i}"])
            return carry, accumulators

        return mosaic.Walk(views, step)

    blocks = stream_blocks(decays=((chunk, keys), gates))
    landings = (0, 1) if not running else (0,) if family.stores_through_registers else ()
    landing = {f"prefix{i}": gates.landing(chunk, keys) for i in landings}
    # The running sum's tiles: its decays, then a sub-chunk's rows of them and their terms; the doubling's
    # five.
    held = (Held(phases=([f32(chunk, keys)], [f32(algebra.SUB, keys)] * 3)) if running
            else Held(phases=([f32(chunk, keys)] * 5,)))
    return launch_kernel(program, call, out_type=(gates.shape(call.heads * width),) * 2,
                         grid=(call.chunks, call.heads, width // keys),
                         grid_names=("chunks", "heads", "blocks"), blocks=blocks, stages=1,
                         live=chunk_liveness(1), steps=1, model=chunk * keys, held=held,
                         staging=_prefix_staging(chunk, keys, running=running), landing=landing,
                         issues=False)(g, taken)


def _running(call: Call) -> bool:
    """Whether `_prefix_halves` sums a key block a column a thread (`_running_prefix`): where the width takes
    the warpgroup's threads' worth of keys and the program's shared memory as its launch lays it (its
    decays' ring slot, which a TMA store lands through, else a landing of its own, the two halves' rows and
    their barriers, `shared_allocation`) fits the target's, else the doubling scan at two sub-chunks' worth of
    keys: an sm120's 99 KB holds no 128 KB program, and the call derived no tiles."""
    if call.width % scan.THREADS:
        return False
    chunk, keys, gates = call.chunk, scan.THREADS, call.gates
    landing = {"prefix0": gates.landing(chunk, keys)} if call.family.stores_through_registers else {}
    _, scope, _, _ = shared_allocation(call, stream_blocks(decays=((chunk, keys), gates)),
                                       _prefix_staging(chunk, keys, running=True), landing, issues=False)
    return resources.shared_bytes(call.family, scope(1)) <= generation.shared_budget(call.target)


def _prefix_staging(chunk: int, keys: int, *, running: bool) -> tuple:
    """`_prefix_halves`' staging: the running sum's two halves' rows `(2, C, keys)`, or the doubling scan's,
    each half's zero rows before its own (`_scan_zeroed`)."""
    return (swizzling.plain((2, chunk if running else 2 * chunk, keys), jnp.float32),)


def _running_prefix(call: Call, rows, decays: jax.Array, keys: int) -> None:
    """A key block's prefix halves of its `decays` `[C, keys]` (`keys` the warpgroup's threads) as a running
    compensated sum down each column, a column a thread (`scan.striped`): the decays staged in `rows`
    `(2, C, keys)`, each token's term (`algebra.prefix_terms`) formed a sub-chunk's rows at a time in a
    row-major layout over them (`_rows_of`), then each row's pair added to the running one, a thread's own
    words (`scan.compensated_columns`; interpreted, `compensated_add` a row at a time), the halves left there.
    The doubling scan staged each of its six steps' pairs through shared memory, 119 thread instructions an
    element, 160 us of an emulated GH200's Kimi Linear forward against fla's cumsum's 40 measured, its stores
    each a bank conflict (M-JS-145c); this, 86 (M-JS-145g); a position here takes 63 compensated adds where
    the doubling took 12, within the prefix's bound either way (the witness's `EXPONENT_BOUND`). The terms
    staged, and the threads met, before they are summed: a product held in registers into a sum was
    contracted into an FMA, its rounding counted twice (`algebra.scanned_prefix`, M-MC-9)."""
    chunk, sub = call.chunk, algebra.SUB
    mosaic.synced(call.interpret)
    plgpu.store(rows.at[0], decays, optimized=False)
    mosaic.synced(call.interpret)
    for start in range(0, chunk, sub):
        # Each thread its own elements of a sub-chunk's rows: read, formed and written back in one layout.
        for half, term in enumerate(algebra.prefix_terms(_rows_of(rows.at[0, pl.ds(start, sub)], keys))):
            plgpu.store(rows.at[half, pl.ds(start, sub)], term, optimized=False)
    mosaic.synced(call.interpret)
    layout = scan.striped(keys)

    if call.interpret:
        # Interpreted, a row at a time in XLA's terms.
        running = (jnp.zeros((keys,), jnp.float32),) * 2
        for t in range(chunk):
            terms = tuple(plgpu.load(rows.at[half, t], layout=layout, optimized=False) for half in (0, 1))
            running = compensated_add(running, terms)
            for half, value in enumerate(running):
                plgpu.store(rows.at[half, t], value, optimized=False)
    else:
        # The warpgroup met when it is done (`compensated_columns`' own barrier).
        scan.compensated_columns(rows)


def _sub_chunk_walk(q, k, hi, lo, beta, taken, call: Call, *, scale: float, products: tuple[str, ...]):
    """The intra walk on a bundle whose products take a sub-chunk's rows (`sub_chunked`), each
    chunk's key blocks from the prefix halves `hi`, `lo` stored (`_prefix_halves`), no scan of its own:
    each step the block's separated prefix (`algebra.separated`) staged for its sub-chunks' references, each
    output's rows gated down and each sub-chunk's keys gated up (`algebra.key_block_pairs`), and
    its products `[SUB, keys] @ [keys, C]` issued into each output's and sub-chunk's sum (`_sum_index`), read
    and stored a sub-chunk's rows at a time once the walk is done (`intra_block_stored`): fla_kda's
    inter-sub-chunk blocks (chunk_kda_fwd_kernel_inter_solve_fused, fla-core 0.5.2 ops/kda/chunk_intra.py)."""
    chunk, width, gates, family = call.chunk, call.width, call.gates, call.family
    scored, outputs, later = "scores" in products, len(products), _later(call)
    keys = key_block(call, functools.partial(_sub_chunk_held, call, outputs=outputs),
                     functools.partial(_sub_chunk_staged, call, products=products),
                     functools.partial(pair_streams, call, queries=scored), contracted=True)
    steps = width // keys
    shrinking = min(abs(scale), 1.0)
    growing = math.copysign(max(abs(scale), 1.0), scale)

    def program(at, q_ref, k_ref, hi_ref, lo_ref, beta_ref, taken_ref, *sums, streams, staging, landing):
        del taken_ref
        n, h = at["chunks"], at["heads"]
        row, span = n * chunk, pl.ds(n * chunk, chunk)
        keyed = h // call.group * width
        views = stream_views(lambda i: (0, i), keys=call.given.views(k_ref, row, chunk, keyed, width),
                             highs=gates.views(hi_ref, row, chunk, h * width, width),
                             lows=gates.views(lo_ref, row, chunk, h * width, width),
                             **({"queries": call.given.views(q_ref, row, chunk, keyed, width)} if scored
                                else {}))
        lefts, raised, separated = staging

        def step(index, slots, carry, accumulators):
            del index
            slots.acquire_all()
            # Each half laid in the family's layout (`Family.tile_of`), as the scan lays them.
            whole = algebra.separated(*(family.tile_of(stream_tile(slots, name, gates).refs[0])
                                        for name in ("highs", "lows")))
            mosaic.synced(call.interpret)
            for half, value in enumerate(whole):
                plgpu.store(separated.at[half], value, optimized=False)
            mosaic.synced(call.interpret)

            def at_token(token):
                """The separated prefix at the chunk's `token`, a column's vectors, from its staged rows."""
                return tuple(family.columns_of(separated.at[half, token]) for half in (0, 1))

            def operand(name):
                return stream_tile(slots, name, call.given).value().astype(jnp.float32)

            def rows(name):
                return intra_rows(call, name, operand, beta_ref.at[h, span])

            issuer = Issuer(call)
            summed = list(accumulators)
            scales = [shrinking if name == "scores" else 1.0 for name in products]
            gated, gated_up = algebra.key_block_pairs([functools.partial(rows, name) for name in products],
                                                      operand("keys"), whole, at_token, chunk=chunk,
                                                      scales=scales)
            held: list = []
            for j in range(later):
                first = (j + 1) * algebra.SUB
                # A sub-chunk's keys staged and published once (`Family.staged_together`), the row sets' gated
                # rows with the first's, its products issued then, running under the next sub-chunk's forming.
                with family.staged_together():
                    keyed = operands.staged(family, gated_up(j + 1), raised[j]).T
                    held = held or [operands.staged(family, value, lefts[r]) for r, value in enumerate(gated)]
                for r, tile in enumerate(held):
                    at = _sum_index(call, r, j)
                    pair = ("pairs", tile.rows(first, algebra.SUB), keyed)
                    summed[at] = issuer.summed([pair], into=summed[at])
            return carry, tuple(summed)

        def read(carry, totals):
            """Each output's sums a sub-chunk's rows at a time, its first sub-chunk's zero (no keys before
            it), the system's columns before each row block's sub-chunk alone (a row whose write gate
            overflows times a zero factor is a NaN past them), the scores scaled past one."""
            del carry
            for r, name in enumerate(products):
                # The first sub-chunk's zero laid as the sums are (`Family.as_tile`): a splat lands strided,
                # where the store's landing takes the sums' tiled layout.
                blocks = [family.as_tile(family.zeros((algebra.SUB, chunk)))]
                for j in range(later):
                    total = totals[_sum_index(call, r, j)]
                    blocks.append(algebra.columns_before(total, (j + 1) * algebra.SUB) if name == "system"
                                  else total * growing)
                for j, block in enumerate(blocks):
                    intra_block_stored(call, name, block, sums[r], landing, row=row + j * algebra.SUB,
                                       head=h)

        return mosaic.Walk(views, step, steps=steps, sums=((algebra.SUB, chunk),) * (outputs * later),
                           read=read)

    blocks = pair_streams(call, keys, queries=scored)
    formed = [dataclasses.replace(call.held, rows=call.rows) if name == "scores" else systems(call)
              for name in products]
    staging, landing = _sub_chunk_staged(call, keys, products=products)
    return launch_kernel(program, call, out_type=tuple(form.shape(call.heads * chunk) for form in formed),
                         grid=(call.chunks, call.heads), grid_names=("chunks", "heads"), blocks=blocks,
                         stages=(1, 2), live=chunk_liveness(5), steps=steps,
                         model=outputs * 2 * later * algebra.SUB * chunk * width,
                         held=_sub_chunk_held(call, keys, outputs), staging=staging, landing=landing,
                         name=_named("recurrence_intra", products), sums=(chunk,) * (outputs * later),
                         swizzles={name: resources.widest_swizzle(dtype, keys)
                                   for name, (_, dtype) in blocks.items()})(q, k, hi, lo, beta, taken)


def _sub_chunk_held(call: Call, block: int, outputs: int = 1) -> Held:
    """`_sub_chunk_walk`'s consumer thread at key block `block`: its sums, a sub-chunk's row of pairs an
    output and later sub-chunk `[SUB, C]`, held across the walk; a step's separated prefix pair, the rows'
    decay down and each output's gated rows `[C, block]`, then a sub-chunk's keys gated up and their masks
    beside a product's fragments (`[SUB, block]` against `[block, C]`)."""
    chunk = call.chunk
    pairing = _pairing_operand(call)
    paired = product_fragments(call, (pairing, algebra.SUB, block), (pairing, block, chunk),
                               rhs_transposed=True)
    carried = [f32(algebra.SUB, chunk)] * (outputs * _later(call))
    return Held(carried=carried, phases=(
        [f32(chunk, block)] * (3 + outputs),
        [f32(chunk, block)] * 3 + masks(chunk, block) + held_tile(pairing, chunk, block) + paired))


def pair_streams(call: Call, keys: int, *, queries: bool) -> dict:
    """The walked streams of a launch forming a chunk's pairs at key block `keys` (`_sub_chunk_walk`'s,
    `own_pairs.intra_own`'s): the keys and both prefix halves a block at a time, and the queries where it
    forms the scores."""
    chunk = call.chunk
    return stream_blocks(keys=((chunk, keys), call.given), highs=((chunk, keys), call.gates),
                         lows=((chunk, keys), call.gates),
                         **({"queries": ((chunk, keys), call.given)} if queries else {}))


def _sub_chunk_staged(call: Call, keys: int, *, products: tuple[str, ...]) -> tuple:
    """`_sub_chunk_walk`'s staging at key block `keys`: each output's gated rows once, each later sub-chunk's
    keys gated up (`algebra.key_block_pairs`), and the step's separated prefix, plain (its rows read
    at the sub-chunks' first tokens); its landings a sub-chunk's row of pairs each output's
    (`intra_block_stored`)."""
    chunk, pairing = call.chunk, _pairing_operand(call)
    landing = {**({"square": dataclasses.replace(call.held, rows=call.rows).landing(algebra.SUB, chunk)}
                  if "scores" in products else {}),
               **({"system": systems(call).landing(algebra.SUB, chunk)} if "system" in products else {})}
    lefts = tuple(staging_tiles(pairing, (chunk, keys))[0] for _ in products)
    raised = staging_tiles(pairing, *((chunk, keys),) * _later(call))
    separated = swizzling.plain((2, chunk, keys), jnp.float32)
    return (lefts, raised, separated), landing


def intra_rows(call: Call, name: str, operand: Callable[[str], jax.Array], beta) -> jax.Array:
    """An intra output's rows of a key block fp32: the scores' the queries; the system's each key scaled by
    its write strength (`beta`, the chunk's `[C]`), its pairs strict; each formed as its products are."""
    if name == "scores":
        return operand("queries")
    return algebra.write_gated(operand("keys"), call.family.rows_of(beta))


def pairing_parts(family: mosaic.Family, held: int) -> int:
    """The parts the factored pairs' fp32 operands take to their products (`algebra.key_block_pairs`) in a
    call holding its values in `held` parts: in a 16-bit call, fp32 itself where the family's products take
    it as TF32 (`Family.takes_tf32`), one product a pair as fla's sub-chunk products take, else two bf16
    parts (16 of their bits, past TF32's); an fp32 call's three bf16 parts."""
    if held > 1:
        return product.EXACT_PARTS
    return 1 if family.takes_tf32 else 2


def _pairing_operand(call: Call) -> Operand:
    """The form the factored pairs' fp32 operands take (`pairing_parts`)."""
    return Operand(call.rows, pairing_parts(call.family, call.held.parts), jnp.dtype(jnp.float32))


