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
    _key_blocks,
    _merges,
    _merging,
    _resident,
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
    solve_persistent,
    staging_tiles,
    stream_blocks,
    stream_tile,
    stream_views,
    walk_stores,
)
from dew.nn.precision import compensated_add


def _intra_held(call: Call, block: int, outputs: int = 1) -> Held:
    """`_intra_walk`'s consumer thread at key block `block` forming `outputs` pairs' sums, which its walk
    holds across the key blocks (`mosaic.Walk.sums`: in registers, but where its family keeps them in memory
    of its own, `Issuer.keeps_sums`): the block's prefix scan (its running pair, a shifted pair and the
    decay); then the factors every output shares beside the block's tiles across its sub-chunks, each
    sub-chunk's operands staged and its products issued in turn (`algebra.key_block_pairs`): its prefix halves
    and their separated pair, the rows' grid, each output's rows gated down, the keys, a sub-chunk's keys
    gated up and an output's rows masked to it as each is formed and staged in its parts, their masks, and a
    product's operands (a row block against a chunk of keys, read transposed). The pairs read their rows' grid
    alone (`algebra.key_block_pairs`), one int32 tile: its columns' too stated 256 and 254 registers for an
    sm100's two intra launches, where ptxas held each at 248 spilling none, least 232 (armada
    recurrence-unused-grid-proof)."""
    chunk = call.chunk
    pairing = _pairing_operand(call)
    paired = product_fragments(call, (pairing, chunk, block), (pairing, block, chunk), rhs_transposed=True)
    carried = [] if call.family.keeps_sums(chunk) else [f32(chunk, chunk)] * outputs
    return Held(carried=carried, phases=(
        [f32(chunk, block)] * 5,
        [f32(chunk, block)] * 9 + masks(chunk, block) + held_tile(pairing, chunk, block) + paired))


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
    head under a decay a key dimension, one launch (`_intra_walk`): the scores as `call.held`, the keys'
    system where it `solves` (`systems`' form, the write-gated keys' pairs, strict), which `_intra_solve`
    inverts, and the decay's prefix halves (`call.gates`, `algebra.prefix`'s pair) from the laid decay `g`
    `[R, HV * Dk]` fp32, which every later launch reads. Each pair of sub-chunks' decay factored
    (`algebra.key_block_pairs`), their products on the tensor cores (fla_kda's intra rows), each sub-chunk's
    own pairs zero (a launch of their own, `own_pairs.intra_own`); the
    scores and the system share every factor in one walk (`_intra_walk`), their sums held across it, where
    `_intra_merges` (wgmma's, tcgen05's); else each output walks a launch of its own."""
    sub = sub_chunked(call)
    products = ("scores", "system") if solves else ("scores",)
    if _sliced(sub):
        # The decay's prefix halves a launch of their own on the call's bundle (fla_kda's chunk_local_cumsum),
        # the walk of the pairs reading them on the sub-chunk's, no scan of its own (`_sub_chunk_walk`).
        hi, lo = _prefix_halves(g, taken, call)
        return (*_sub_chunk_walk(q, k, hi, lo, beta, taken, sub, scale=scale, products=products), hi, lo)
    if solves and not _intra_merges(call):
        scores, hi, lo = _intra_walk(q, k, g, beta, taken, call, scale=scale, products=("scores",))
        (system,) = _intra_walk(q, k, g, beta, taken, call, scale=scale, products=("system",))
        walked = (scores, system)
    else:
        *walked, hi, lo = _intra_walk(q, k, g, beta, taken, call, scale=scale, products=products)
    return (*walked, hi, lo)


def sub_chunked(call: Call) -> Call:
    """`call` on the bundle whose products take a sub-chunk's rows (`bundles.layouts` at `algebra.SUB`
    rows: mma.sync's on every target, an H100's and a B200's among them, as a decode step's 8 rows take it),
    the intra walk's (`_sliced`): a sub-chunk's row of pairs a product, fla_kda's inter-sub-chunk blocks
    (chunk_kda_fwd_kernel_inter_solve_fused). On warpgroup MMA's 64 rows each sub-chunk's product took a
    whole chunk's rows masked to its own, 2.3 times fla's inter-sub-chunk launch on a GH200 and 13 times
    Dew's port of it on a B200 (M-JS-144-gh200: 330 + 87 us against 142 with the solve; js106c: 712 + 664
    against 102)."""
    # Off a GPU no bundle claims the target: the call's own, interpreted.
    found = bundles.layouts(call.target.generation, bundle_dtype(call.given.dtype), algebra.SUB, scoped=True)
    family = found[0] if found else call.family
    if family.rows <= algebra.SUB:
        # Its warps along a sub-chunk's row of pairs' columns, its sums' layout `[SUB, C]`'s (`specialized`).
        family = family.specialized(algebra.SUB, call.chunk)
    return call if family == call.family else dataclasses.replace(call, family=family)


def _sliced(call: Call) -> bool:
    """Whether `call`'s products take a sub-chunk's rows (`algebra.key_block_pairs`' `sliced`)."""
    return call.family.rows <= algebra.SUB


def _later(call: Call) -> int:
    """The sub-chunks past a chunk's first, each a row of pairs (`algebra.key_block_pairs`' products)."""
    return call.chunk // algebra.SUB - 1


def _named(launch: str, products: tuple[str, ...]) -> str:
    """The name of `launch`'s walk of `products`: its own where it walks every output (the scores and the
    system), else `<launch>_<output>`, so a walk of one output is a kernel of its own name beside the walk of
    both, which a training call's backward and forward each launch (a kernel's name is its identity,
    `report.named`: Kimi's sm90 training call ran two `recurrence_intra` bodies under one name)."""
    return launch if len(products) > 1 else "_".join((launch, *products))


def _intra_merges(call: Call) -> bool:
    """Whether the scores and the system take one intra walk (`_intra_walk`): where the family's products
    read their operands from shared memory (wgmma's, tcgen05's), at a key block at which a multiprocessor
    holds a program of it, both sums held across the walk. Where they read registers (mma.sync's), each
    output walks a launch of its own: one walk's sums through shared memory beside the fragments took 1.736
    ms of an A100's Kimi Linear forward against the two launches' 0.762 and 0.717 (M-JS-17, W33-a100-1), the
    forward 2.754 ms against 2.487."""
    if call.family.operands_in_registers:
        return False
    resident = functools.partial(_resident, call, functools.partial(_intra_held, call, outputs=2),
                                 functools.partial(_intra_staged, call, products=("scores", "system")),
                                 functools.partial(_intra_streamed, call, queries=True))
    return any(resident(block) for block in _key_blocks(call, contracted=True))


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


def intra_sums_stored(call: Call, products: tuple[str, ...], totals: tuple, sums, landing: dict, *, row,
                       head) -> None:
    """The intra walk's sums once read, each output's in its form through its landing: the scores as the held
    operand's, the system as `systems`' (fp32, as the sums are)."""
    chunk = call.chunk
    held = dataclasses.replace(call.held, rows=call.rows)
    for name, total, ref in zip(products, totals, sums, strict=True):
        form, smem = (held, landing["square"]) if name == "scores" else (systems(call), landing["system"])
        form.store(ref, total, row, chunk, head * chunk, chunk, family=call.family, smem=smem)


def intra_block_stored(call: Call, name: str, block, ref, landing: dict, *, row, head) -> None:
    """A sliced intra walk's sub-chunk row block `[SUB, C]` of output `name` at `row`, as
    `intra_sums_stored` stores a whole chunk's."""
    chunk = call.chunk
    form, smem = ((dataclasses.replace(call.held, rows=call.rows), landing["square"]) if name == "scores"
                  else (systems(call), landing["system"]))
    form.store(ref, block, row, algebra.SUB, head * chunk, chunk, family=call.family, smem=smem)


def _sum_index(call: Call, output: int, later: int) -> int:
    """`_sub_chunk_walk`'s sum of output `output`'s and the `later`-th sub-chunk past the first's."""
    return output * _later(call) + later


def _intra_walk(q, k, g, beta, taken, call: Call, *, scale: float, products: tuple[str, ...]):
    """The intra launch's walk of each chunk's key blocks (`key_block`, the block a multiprocessor holds most
    programs of): each step the block's prefix halves from its decay (`algebra.scanned_prefix`, staged in a
    padded shared tile whose rows the sub-chunks' references read), stored for the later launches, and the
    block's pairs' products issued into the sums the walk holds (`mosaic.Walk.sums`), read and stored once
    after it: the scores (the queries' pairs, scaled) and, under the delta rule, the keys' system
    (the write-gated keys' pairs, strict), their factors formed once for both (`algebra.key_block_pairs`), and
    each sub-chunk's own pairs each their own decay (`algebra.key_diagonal_pairs`).
    Its prefix an XLA scan of its own, thirteen launches took 0.30 ms of Kimi Linear's 1.57 ms forward on an
    H100; the scores and the system a launch each, each scanned the prefix and gated its keys again, 0.42
    and 0.40 ms of its 1.45 (kda-batch-h100-7e3a5fb48)."""
    chunk = call.chunk
    width, gates, family = call.width, call.gates, call.family
    # The scores' launch stores the prefix halves; a launch of the system alone reads no queries.
    scored = "scores" in products
    outputs = len(products)
    keys = key_block(call, functools.partial(_intra_held, call, outputs=outputs),
                     functools.partial(_intra_staged, call, products=products),
                     functools.partial(_intra_streamed, call, queries=scored), contracted=True)
    steps = width // keys
    # The scores' scale where each part stays in range: its magnitude up to one on the gated rows, which
    # shrinks every product and so every sum held across the key blocks (the pairs of two unscaled blocks
    # near fp32's largest pass it); past one, and its sign, on the read sums, whose products a gated-up key
    # may attenuate after a row's scale would have overflowed it (a query of 1e30 at a scale of 1e10).
    shrinking = min(abs(scale), 1.0)
    growing = math.copysign(max(abs(scale), 1.0), scale)

    def program(at, q_ref, k_ref, g_ref, beta_ref, taken_ref, *outs, streams, staging, landing):
        n, h = at["chunks"], at["heads"]
        row, span = n * chunk, pl.ds(n * chunk, chunk)
        keyed = h // call.group * width
        views = stream_views(lambda i: (0, i), keys=call.given.views(k_ref, row, chunk, keyed, width),
                             decays=gates.views(g_ref, row, chunk, h * width, width),
                             **({"queries": call.given.views(q_ref, row, chunk, keyed, width)} if scored
                                else {}))
        lefts, raised, scanned = staging
        sums, prefixes = outs[:outputs], outs[outputs:]
        # The prefix halves' stores left in flight through their own landings, each waited for by the next
        # step's store of its half (`walk_stores`): stored and waited out each step, the stores' waits took
        # 9% of a B200's Kimi Linear intra walk (an emulated B200's line map at 3fd15d5978b).
        before_store = walk_stores(family, 2)

        def step(index, slots, carry, accumulators):
            slots.acquire_all()
            prefix = _intra_prefix(call, scanned, stream_tile(slots, "decays", gates).value(), keys)
            for i, (half, ref) in enumerate(zip(prefix if scored else (), prefixes, strict=True)):
                before_store()
                gates.store(ref, half, row, chunk, h * width + index * keys, keys, family=family,
                            smem=landing[f"prefix{i}"], wait=False)

            # The prefix separated once (`algebra.separated`) and staged where the scan left it, each
            # sub-chunk boundary's vectors read from it (`at_token`), not separated again at every read.
            whole = algebra.separated(*prefix)
            _staged_rows(call, scanned, whole)

            def at_token(token):
                """The separated prefix at the chunk's `token`, a column's vectors, from its staged rows."""
                return tuple(family.columns_of(scanned.at[half, chunk + token]) for half in (0, 1))

            def operand(name):
                return stream_tile(slots, name, call.given).value().astype(jnp.float32)

            def rows(name):
                return intra_rows(call, name, operand, beta_ref.at[h, span])

            def staged(value, s, r):
                """A sub-chunk's operand staged in its own tiles (`_intra_staged`): its keys gated up, or row
                set `r`'s rows gated down."""
                return operands.staged(family, value, raised[s - 1] if r is None else lefts[r][s - 1])

            issuer = Issuer(call)
            for formed in algebra.key_block_pairs([functools.partial(rows, name) for name in products],
                                                  operand("keys"), whole, at_token, staged, chunk=chunk,
                                                  scales=[shrinking if name == "scores" else 1.0
                                                          for name in products]):
                # A sub-chunk's operands staged and published once (`Family.staged_together`), its products
                # issued then, running under the next sub-chunk's forming.
                with family.staged_together():
                    found = formed()
                accumulators = tuple(issuer.summed([product], into=acc)
                                     for product, acc in zip(found, accumulators, strict=True))
            return carry, accumulators

        def read(carry, totals):
            """The sums once the walk is done, the system masked once (a row whose write gate overflows
            times a zero factor is a NaN there, which its strict mask clears), and stored."""
            del carry
            masked = tuple(algebra.strictly_lower(total) if name == "system" else total * growing
                           for name, total in zip(products, totals, strict=True))
            intra_sums_stored(call, products, masked, sums, landing, row=row, head=h)

        return mosaic.Walk(views, step, steps=steps, prologue=lambda: _scan_zeroed(call, scanned, keys),
                           sums=((chunk, chunk),) * outputs, read=read)

    blocks = _intra_streamed(call, keys, queries=scored)
    formed = [dataclasses.replace(call.held, rows=call.rows) if name == "scores" else systems(call)
              for name in products]
    out_type = (*(form.shape(call.heads * chunk) for form in formed),
                *((gates.shape(call.heads * width),) * 2 if scored else ()))
    # Each output's products a whole chunk of rows against a whole chunk of keys, the model's own pairs'.
    model = outputs * 2 * chunk * chunk * width
    staging, landing = _intra_staged(call, keys, products=products)
    # Its rings as deep as its residency holds them (`resources.plan`): one deep, a B200's Kimi Linear walk
    # waited on each key block's loads at two programs a multiprocessor, 24.5% of its warps active and
    # long_scoreboard 6.2 cycles an issued instruction, 819 us against fla's intra rows' 589 (M-JS-118).
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=blocks, stages=(1, 2),
                         live=chunk_liveness(4), steps=steps, model=model,
                         held=_intra_held(call, keys, outputs), staging=staging, landing=landing,
                         name=_named("recurrence_intra", products),
                         sums=(chunk,) * outputs,
                         # Each stream staged at its key block's row: a tile slices only whole swizzle rows.
                         swizzles={name: resources.widest_swizzle(dtype, keys)
                                   for name, (_, dtype) in blocks.items()})(q, k, g, beta, taken)


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
            if running:
                prefix = _running_prefix(call, rows, decays, keys)
            else:
                _scan_zeroed(call, rows, keys)
                prefix = _intra_prefix(call, rows, decays, keys)
            # The running sum's halves through its decays' slot in turn, read and staged before (each store
            # waited; the operand's form and tile, a landing's own), the doubling's through their landings.
            for i, (half, ref) in enumerate(zip(prefix, (hi_ref, lo_ref), strict=True)):
                gates.store(ref, half, row, chunk, column, keys, family=family,
                            smem=tile.refs[0] if running else landing[f"prefix{i}"])
            return carry, accumulators

        return mosaic.Walk(views, step)

    blocks = stream_blocks(decays=((chunk, keys), gates))
    landing = {} if running else {f"prefix{i}": gates.landing(chunk, keys) for i in (0, 1)}
    # The running sum's tiles: its decays, then a sub-chunk's rows of them and their terms; the doubling's
    # five.
    held = (Held(phases=([f32(chunk, keys)], [f32(algebra.SUB, keys)] * 3)) if running
            else Held(phases=([f32(chunk, keys)] * 5,)))
    return launch_kernel(program, call, out_type=(gates.shape(call.heads * width),) * 2,
                         grid=(call.chunks, call.heads, width // keys),
                         grid_names=("chunks", "heads", "blocks"), blocks=blocks, stages=1,
                         live=chunk_liveness(1), steps=1, model=chunk * keys, held=held,
                         staging=(swizzling.plain((2, chunk if running else 2 * chunk, keys), jnp.float32),),
                         landing=landing)(g, taken)


def _running(call: Call) -> bool:
    """Whether `_prefix_halves` sums a key block a column a thread (`_running_prefix`): where the width takes
    the warpgroup's threads' worth of keys and the program's three `[C, keys]` fp32 tiles (its decays' ring
    slot, which its stores land through, and the two halves' rows) fit the target's shared memory, else the
    doubling scan at two sub-chunks' worth of keys: an sm120's 99 KB holds no 128 KB program, and the call
    derived no tiles."""
    tiles = 3 * call.chunk * scan.THREADS * jnp.dtype(jnp.float32).itemsize
    return call.width % scan.THREADS == 0 and tiles <= generation.shared_budget(call.target)


def _running_prefix(call: Call, rows, decays: jax.Array, keys: int) -> tuple:
    """A key block's prefix halves of its `decays` `[C, keys]` (`keys` the warpgroup's threads) as a running
    compensated sum down each column, a column a thread (`scan.striped`): the decays staged in `rows`
    `(2, C, keys)`, each token's term (`algebra.prefix_terms`) formed a sub-chunk's rows at a time in a
    row-major layout over them (`_rows_of`), then each row's pair added to the running one (`compensated_add`)
    and stored over its terms, the halves returned in the accumulator's layout. The doubling scan staged each
    of its six steps' pairs through shared memory, 119 thread instructions an element, 160 us of an emulated
    GH200's Kimi Linear forward against fla's cumsum's 40 measured, its stores each a bank conflict
    (M-JS-145c); a position here takes 63 compensated adds where the doubling took 12, within the prefix's
    bound either way (the witness's `EXPONENT_BOUND`). The terms staged, and the threads met, before they are
    summed: a product held in registers into a sum was contracted into an FMA, its rounding counted twice
    (`algebra.scanned_prefix`, M-MC-9)."""
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

    def row(t) -> tuple:
        return tuple(plgpu.load(rows.at[half, t], layout=layout, optimized=False) for half in (0, 1))

    # Unrolled, each row's load issued a row ahead of its add: one program a multiprocessor (its shared
    # memory's) waited on every row's load in turn, 58% of its stalls (M-JS-145e, an emulated GH200).
    zero = plgpu.layout_cast(jnp.zeros((keys,), jnp.float32), layout)
    running, ahead = (zero, zero), row(0)
    for t in range(chunk):
        terms, ahead = ahead, row(t + 1) if t + 1 < chunk else None
        running = compensated_add(running, terms)
        for half, value in enumerate(running):
            plgpu.store(rows.at[half, t], value, optimized=False)
    mosaic.synced(call.interpret)
    return tuple(call.family.tile_of(rows.at[half]) for half in (0, 1))


def _sub_chunk_walk(q, k, hi, lo, beta, taken, call: Call, *, scale: float, products: tuple[str, ...]):
    """The intra walk on a bundle whose products take a sub-chunk's rows (`sub_chunked`, `_sliced`), each
    chunk's key blocks from the prefix halves `hi`, `lo` stored (`_prefix_halves`), no scan of its own:
    each step the block's separated prefix (`algebra.separated`) staged for its sub-chunks' references, each
    output's rows gated down and each sub-chunk's keys gated up (`algebra.key_block_pairs`' `sliced`), and
    its products `[SUB, keys] @ [keys, C]` issued into each output's and sub-chunk's sum (`_sum_index`), read
    and stored a sub-chunk's rows at a time once the walk is done (`intra_block_stored`): fla_kda's
    inter-sub-chunk blocks (chunk_kda_fwd_kernel_inter_solve_fused, fla-core 0.5.2 ops/kda/chunk_intra.py)."""
    chunk, width, gates, family = call.chunk, call.width, call.gates, call.family
    scored, outputs, later = "scores" in products, len(products), _later(call)
    keys = key_block(call, functools.partial(_sub_chunk_held, call, outputs=outputs),
                     functools.partial(_sub_chunk_staged, call, products=products),
                     functools.partial(_sub_chunk_streamed, call, queries=scored), contracted=True)
    steps = width // keys
    shrinking = min(abs(scale), 1.0)
    growing = math.copysign(max(abs(scale), 1.0), scale)

    def program(at, q_ref, k_ref, hi_ref, lo_ref, beta_ref, taken_ref, *sums, streams, staging, landing):
        del taken_ref
        n, h = at["chunks"], at["heads"]
        row, span = n * chunk, pl.ds(n * chunk, chunk)
        keyed = h // call.group * width
        views = stream_views(lambda i: (0, i), keys=call.given.views(k_ref, row, chunk, keyed, width),
                             hi=gates.views(hi_ref, row, chunk, h * width, width),
                             lo=gates.views(lo_ref, row, chunk, h * width, width),
                             **({"queries": call.given.views(q_ref, row, chunk, keyed, width)} if scored
                                else {}))
        lefts, raised, separated = staging

        def step(index, slots, carry, accumulators):
            del index
            slots.acquire_all()
            # Each half laid in the family's layout (`Family.tile_of`), as the scan lays them.
            whole = algebra.separated(*(family.tile_of(stream_tile(slots, name, gates).refs[0])
                                        for name in ("hi", "lo")))
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

            def staged(value, s, r):
                """The keys gated up to sub-chunk `s` in their tile, or row set `r`'s rows gated down."""
                return operands.staged(family, value, raised[s - 1] if r is None else lefts[r])

            issuer = Issuer(call)
            summed = list(accumulators)
            for j, formed in enumerate(algebra.key_block_pairs(
                    [functools.partial(rows, name) for name in products], operand("keys"), whole, at_token,
                    staged, chunk=chunk, scales=[shrinking if name == "scores" else 1.0 for name in products],
                    sliced=True)):
                with family.staged_together():
                    found = formed()
                for r, pair in enumerate(found):
                    at = _sum_index(call, r, j)
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

    blocks = _sub_chunk_streamed(call, keys, queries=scored)
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


def _sub_chunk_streamed(call: Call, keys: int, *, queries: bool) -> dict:
    """`_sub_chunk_walk`'s walked streams at key block `keys`: the keys and both prefix halves a block at a
    time, and the queries where it forms the scores."""
    chunk = call.chunk
    return stream_blocks(keys=((chunk, keys), call.given), hi=((chunk, keys), call.gates),
                         lo=((chunk, keys), call.gates),
                         **({"queries": ((chunk, keys), call.given)} if queries else {}))


def _sub_chunk_staged(call: Call, keys: int, *, products: tuple[str, ...]) -> tuple:
    """`_sub_chunk_walk`'s staging at key block `keys`: each output's gated rows once, each later sub-chunk's
    keys gated up (`algebra.key_block_pairs`' `sliced`), and the step's separated prefix, plain (its rows read
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


def _intra_streamed(call: Call, keys: int, *, queries: bool) -> dict:
    """`_intra_walk`'s walked streams at key block `keys`: the keys and the decay a block at a time, and the
    queries where it forms the scores."""
    chunk = call.chunk
    return stream_blocks(keys=((chunk, keys), call.given), decays=((chunk, keys), call.gates),
                         **({"queries": ((chunk, keys), call.given)} if queries else {}))


def _intra_staged(call: Call, keys: int, *, products: tuple[str, ...]) -> tuple:
    """`_intra_walk`'s staging tiles and landings at key block `keys` forming `products`: each sub-chunk past
    the first's operands of its products (`algebra.key_block_pairs`), every row set's rows gated down and the
    keys gated up, a `[C, keys]` tile a part each (`_pairing_operand`); the prefix scan's halves below as
    many zero rows (`algebra.scanned_prefix`'s shifts); and the stores' landings: the scores' square and a
    block of each prefix half where it forms the scores (`walk_stores`), the system's where it forms the
    system."""
    chunk = call.chunk
    landing = {**({"square": dataclasses.replace(call.held, rows=call.rows).landing(chunk, chunk),
                   **{f"prefix{i}": call.gates.landing(chunk, keys) for i in (0, 1)}}
                  if "scores" in products else {}),
               **({"system": systems(call).landing(chunk, chunk)} if "system" in products else {})}
    later = chunk // algebra.SUB - 1
    lefts = tuple(staging_tiles(_pairing_operand(call), *((chunk, keys),) * later) for _ in products)
    raised = staging_tiles(_pairing_operand(call), *((chunk, keys),) * later)
    # Read at row offsets no tiling divides (the scan's shifts, the sub-chunks' references), so plain.
    # Unpadded: a tile read in the accumulator's layout takes a row stride a whole number of its tiling's
    # columns (Mosaic GPU's layout inference refused 24 at a 16-column tiling and 80 at a 64-column one,
    # jsf-kdaintra5 and 9), a multiple of 32 words at 32 columns or more, so no stride it takes moves a
    # fragment store's rows off one another's banks.
    scanned = swizzling.plain((2, 2 * chunk, keys), jnp.float32)
    return (lefts, raised, scanned), landing


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


