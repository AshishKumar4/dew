"""The recurrence template's intra launches under a decay a key dimension (Kimi delta attention's): each
chunk's pairs within it, the scores and the keys' system, walked a key block at a time on the mainloop's
skeleton (`mainloop.launch_kernel`), their factors formed once for both (the algebra's `key_block_pairs`),
and the system's inverse merged from its sub-chunks' blocks by forward substitution after the walk, as
fla_kda's fused inter-sub-chunk solve; each sub-chunk's own pairs, and their system's inverse, are
`own_pairs`'. A seam of the mainloop of its own: these launches read its primitives, and nothing of the
mainloop reads them."""

import dataclasses
import functools
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.mosaic.gpu import utils
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import arith, memref

from dew.nn.kernels import generation, recurrence_algebra as algebra
from dew.nn.kernels.templates import bundles, mosaic, operands, product, resources, scan, swizzling
from dew.nn.kernels.templates.operands import Issuer
from dew.nn.kernels.templates.recurrence.mainloop import (
    Call,
    Held,
    Operand,
    _merged,
    building,
    bundle_dtype,
    chunk_liveness,
    clamped,
    f32,
    held_tile,
    key_block,
    launch_kernel,
    limb_blocks,
    limb_tiles,
    limb_views,
    lowered_once,
    masks,
    product_fragments,
    shared_allocation,
    staging_tiles,
    stream_blocks,
    stream_tile,
    stream_views,
)
from dew.nn.precision import compensated_add


def systems(call: Call) -> Operand:
    """The form a chunk's keys' system takes in fp32, as fla's intra rows store their `Akk`: its pairs'
    backward's (`key_backward`)."""
    return Operand(call.rows, 1, jnp.dtype(jnp.float32))


def placed_form(call: Call) -> Operand:
    """The form the own pairs' scores (`own_pairs.intra_own`'s) take to the walk's product placing them on
    their diagonal blocks (`_sub_chunk_walk`'s 'place', against an exact identity): the held scores' where
    16-bit (a part times one, exact), else three bf16 parts, which an fp32 value is exactly (a TF32 operand
    rounds it)."""
    held = dataclasses.replace(call.held, rows=call.rows)
    if held.stored != jnp.float32:
        return held
    return Operand(call.rows, product.bf16_part_count(jnp.float32, accurate=True), jnp.dtype(jnp.float32))


def inverse_form(call: Call) -> Operand:
    """The form the own pairs' inverted blocks (`own_pairs.intra_own`'s) and the walk's inverse take to the
    solve's products (`_sub_chunk_walk`): the doubling's merges' (`mainloop._merged`), on the walk's bundle
    (`sub_chunked`)."""
    return _merged(sub_chunked(call))


@building(lambda launch: ("recurrence_intra", "recurrence_intra_scores"))
def _intra(q, k, prefix: tuple, beta, taken, own: tuple, call: Call, *, scale: float, solves: bool):
    """`(Aqk [R, HV * C], (I + A)^-1 less its unit diagonal [R, HV * C] where the call `solves`)` of each
    chunk and value head under a decay a key dimension, as `call.held`, from the decay's `prefix` (`_prefix`'s
    limbs) and each sub-chunk's own blocks `own` (`own_pairs.intra_own`'s, compact: the scores' and the
    system's inverted), in one walk sharing every factor (`_sub_chunk_walk`): each pair of sub-chunks' decay
    factored (`algebra.key_block_pairs`), their products on the tensor cores (fla_kda's intra rows), then each
    sub-chunk's own scores placed on its diagonal and the system's inverse merged from its blocks, fla_kda's
    chunk_kda_fwd_kernel_inter_solve_fused (fla-core 0.5.2 ops/kda/chunk_intra.py)."""
    products = ("scores", "system") if solves else ("scores",)
    return _sub_chunk_walk(q, k, prefix, beta, taken, own, sub_chunked(call), scale=scale, products=products)


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
    """The intra walk's sub-chunk row block `[SUB, C]` of output `name` (the scores, or the system's inverse)
    at `row`, a sub-chunk's rows of the chunk's square, as `call.held`, each through its own landing."""
    chunk = call.chunk
    form = dataclasses.replace(call.held, rows=call.rows)
    form.store(ref, block, row, algebra.SUB, head * chunk, chunk, family=call.family,
               smem=landing["square" if name == "scores" else "inverse"])


def _sum_index(call: Call, output: int, later: int) -> int:
    """`_sub_chunk_walk`'s sum of output `output`'s and the `later`-th sub-chunk past the first's."""
    return output * _later(call) + later


@lowered_once
def _prefix(g, taken, call: Call):
    """`[R, HV * Dk]` fp32: the laid decay `g`'s prefix within each chunk (`call.gates`), the compensated
    sum's high half (`algebra.prefix`'s pair rounded to one fp32), a program a chunk, value head and key block
    (`_running`), each its block's one load, scan and store: a running compensated sum down each column where
    the block fills the warpgroup's threads (`_running_prefix`), else the doubling scan (`_intra_prefix`).
    fla_kda's chunk_local_cumsum, a launch the sub-chunk walk and every later launch read (`_sub_chunk_walk`):
    a tuple of the halves its call keeps (`Call.prefix_limbs`), each `[R, HV * Dk]`. Where its layer bounds
    its decays, the high half alone, as fla_kda stores one fp32 cumulative sum: both halves stored, every
    launch reading the decay read twice fla's bytes of it, where the rounded sum alone holds check 1 against
    fla at every Kimi delta attention cell and stability regime (scratch jsf-prefixC); a plain fp32
    cumulative sum, no compensation, does not (its final memory 9.4 to 9.9 times fla's distance from float64
    at the stability regimes, scratch jsf-prefixA)."""
    chunk, width, gates, family = call.chunk, call.width, call.gates, call.family
    running = _running(call)
    keys = scan.THREADS if running else math.gcd(width, 2 * algebra.SUB)
    # The running sum's rows a block at a time, its running pairs carried between (`_running_prefix`).
    rows_a_step = _running_rows(chunk) if running else chunk
    steps = chunk // rows_a_step
    limbs = call.prefix_limbs

    def program(at, g_ref, taken_ref, *prefix_refs, streams, staging, landing):
        del taken_ref
        n, h, b = at["chunks"], at["heads"], at["blocks"]
        row, column = n * chunk, h * width + b * keys
        views = stream_views(decays=gates.views(g_ref, row, chunk, column, keys))
        rows, *carried = staging

        def zeroed():
            # Each column's running pair zero as the chunk starts.
            if call.interpret:
                plgpu.store(carried[0], jnp.zeros((2, keys), jnp.float32), optimized=False)
            else:
                scan.zeroed_columns(carried[0])

        def step(index, slots, carry, accumulators):
            slots.acquire_all()
            tile = stream_tile(slots, "decays", gates)
            found = None
            # A clamped call's decays clamped as they are read (`Call.clamped`): in XLA, a pass over
            # them, 91 us of Kimi Linear's forward on an A100 (M-JS-226).
            floored = clamped if call.clamped else (lambda decays: decays)
            if running:
                # Row-major from the slot, plain (`_plain_decays`), as a step's rows are fewer than a
                # warpgroup accumulator's tile takes (the gates one fp32 part).
                _running_prefix(call, rows, carried[0], floored(_rows_of(tile.refs[0], keys)), keys)
            else:
                _scan_zeroed(call, rows, keys)
                found = _intra_prefix(call, rows, floored(tile.value()), keys)
            for i, ref in enumerate(prefix_refs):
                # A running sum's halves read as each is stored, one tile held at a time: both at once
                # spilled the program's 128 registers at two programs a multiprocessor (M-JS-145g). Row-major,
                # as a step's rows are fewer than a warpgroup accumulator's tile takes, and stored through
                # their own plain rows (fp32, the gates' form: a family's swizzled slot took no row-major
                # tile, an sm100's); the doubling's through their own landings.
                half = _rows_of(rows.at[i], keys) if found is None else found[i]
                gates.store(ref, half, row + index * rows_a_step, rows_a_step, column, keys, family=family,
                            smem=rows.at[i] if running else landing[f"prefix{i}"])
            return carry, accumulators

        return mosaic.Walk(views, step, steps=steps, prologue=zeroed if running else None)

    blocks = stream_blocks(decays=((rows_a_step, keys), gates))
    landing = {} if running else {f"prefix{i}": gates.landing(rows_a_step, keys) for i in range(limbs)}
    # The running sum's tiles: a block's decays, then a sub-chunk's rows of them and their terms; the
    # doubling's five.
    held = (Held(phases=([f32(rows_a_step, keys)], [f32(algebra.SUB, keys)] * 3)) if running
            else Held(phases=([f32(chunk, keys)] * 5,)))
    return launch_kernel(program, call, out_type=(gates.shape(call.heads * width),) * limbs,
                         grid=(call.chunks, call.heads, width // keys),
                         grid_names=("chunks", "heads", "blocks"), blocks=blocks, stages=1,
                         live=chunk_liveness(1), steps=steps, model=chunk * keys, held=held,
                         staging=_prefix_staging(chunk, keys, running=running), landing=landing,
                         issues=False, swizzles=_plain_decays(running=running))(g, taken)


def _plain_decays(*, running: bool) -> dict[str, int] | None:
    """The running sum's decays' slot unswizzled (`swizzling.UNSWIZZLED`): its rows read into registers alone,
    a step's fewer than a warpgroup accumulator's tile, which a family's swizzled slot took in no layout (an
    sm80's and an sm100's, jsf-kdan-bis); the doubling's the family's own."""
    return {"decays0": swizzling.UNSWIZZLED} if running else None


def _running_rows(chunk: int) -> int:
    """The rows `_prefix`'s running sum takes a step: half a chunk's, a whole number of sub-chunks,
    its running pairs carried between. A chunk's rows at once staged 98 KB a program, two programs an
    emulated GH200's multiprocessor, the launch latency-bound (54% of its warp time a long scoreboard, its
    DRAM 40% busy, M-JS-170), one an A100's."""
    return max(algebra.SUB, chunk // 2)


def _running(call: Call) -> bool:
    """Whether `_prefix` sums a key block a column a thread (`_running_prefix`): where the width takes
    the warpgroup's threads' worth of keys and the program's shared memory as its launch lays it (its
    decays' ring slot, which a TMA store lands through, else a landing of its own, the two halves' rows and
    their barriers, `shared_allocation`) fits the target's, else the doubling scan at two sub-chunks' worth of
    keys: an sm120's 99 KB holds no 128 KB program, and the call derived no tiles."""
    if call.width % scan.THREADS:
        return False
    chunk, keys, gates = call.chunk, scan.THREADS, call.gates
    rows = _running_rows(chunk)
    _, scope, _, _ = shared_allocation(call, stream_blocks(decays=((rows, keys), gates)),
                                       _prefix_staging(chunk, keys, running=True), {}, issues=False,
                                       swizzles=_plain_decays(running=True))
    return resources.shared_bytes(call.family, scope(1)) <= generation.shared_budget(call.target)


def _prefix_staging(chunk: int, keys: int, *, running: bool) -> tuple:
    """`_prefix`'s staging: the running sum's two halves' rows a step's `(2, _running_rows, keys)` and
    each column's running pair `(2, keys)`, or the doubling scan's, each half's zero rows before its own
    (`_scan_zeroed`)."""
    if running:
        return (swizzling.plain((2, _running_rows(chunk), keys), jnp.float32),
                swizzling.plain((2, keys), jnp.float32))
    return (swizzling.plain((2, 2 * chunk, keys), jnp.float32),)


def _running_prefix(call: Call, rows, running, decays: jax.Array, keys: int) -> None:
    """A key block's prefix halves of a step's `decays` `[rows, keys]` (`keys` the warpgroup's threads) as a
    running compensated sum down each column from its pair in `running` `(2, keys)`, left there for the
    next step's rows, a column a thread (`scan.striped`): the decays staged in `rows` `(2, rows, keys)`, each
    token's term (`algebra.prefix_terms`) formed a sub-chunk's rows at a time in a row-major layout over them
    (`_rows_of`), then each row's pair added to the running one, a thread's own words
    (`scan.compensated_columns`; interpreted, `compensated_add` a row at a time), the halves left there.
    The doubling scan staged each of its six steps' pairs through shared memory, 119 thread instructions an
    element, 160 us of an emulated GH200's Kimi Linear forward against fla's cumsum's 40 measured, its stores
    each a bank conflict (M-JS-145c); this, 86 (M-JS-145g); a position here takes 63 compensated adds where
    the doubling took 12, within the prefix's bound either way (the witness's `EXPONENT_BOUND`). The terms
    staged, and the threads met, before they are summed: a product held in registers into a sum was
    contracted into an FMA, its rounding counted twice (`algebra.scanned_prefix`, M-MC-9)."""
    chunk, sub = decays.shape[0], algebra.SUB
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
        pair = tuple(plgpu.load(running.at[half], layout=layout, optimized=False) for half in (0, 1))
        for t in range(chunk):
            terms = tuple(plgpu.load(rows.at[half, t], layout=layout, optimized=False) for half in (0, 1))
            pair = compensated_add(pair, terms)
            for half, value in enumerate(pair):
                plgpu.store(rows.at[half, t], value, optimized=False)
        for half, value in enumerate(pair):
            plgpu.store(running.at[half], value, optimized=False)
    else:
        # The warpgroup met when it is done (`compensated_columns`' own barrier).
        scan.compensated_columns(rows, running)


def _sub_chunk_walk(q, k, prefix, beta, taken, own: tuple, call: Call, *, scale: float,
                    products: tuple[str, ...]):
    """The intra walk on a bundle whose products take a sub-chunk's rows (`sub_chunked`), each
    chunk's key blocks from the prefix stored (`_prefix`), no scan of its own:
    each step the block's separated prefix (`algebra.separated`) staged for its sub-chunks' references, each
    output's rows gated down and each sub-chunk's keys gated up (`algebra.key_block_pairs`), and
    its products `[SUB, keys] @ [keys, C]` issued into each output's and sub-chunk's sum (`_sum_index`), read
    a sub-chunk's rows at a time once the walk is done (`_read_out`): fla_kda's inter-sub-chunk blocks
    (chunk_kda_fwd_kernel_inter_solve_fused, fla-core 0.5.2 ops/kda/chunk_intra.py), each sub-chunk's own
    blocks `own` (`own_pairs.intra_own`'s, the scores' and the system's inverses) read once a chunk."""
    chunk, width, family = call.chunk, call.width, call.family
    scored, outputs, later = "scores" in products, len(products), _later(call)
    placed, inverted = placed_form(call), inverse_form(call)
    keys = key_block(call, functools.partial(_sub_chunk_held, call, outputs=outputs),
                     functools.partial(_sub_chunk_staged, call, products=products),
                     functools.partial(pair_streams, call, queries=scored), contracted=True)
    steps = width // keys
    shrinking = min(abs(scale), 1.0)
    growing = math.copysign(max(abs(scale), 1.0), scale)

    def program(at, q_ref, k_ref, *refs, streams, staging, landing):
        prefix_refs, (beta_ref, _taken_ref, *rest) = refs[:len(prefix)], refs[len(prefix):]
        own_refs, sums = rest[:len(own)], rest[len(own):]
        n, h = at["chunks"], at["heads"]
        row, span = n * chunk, pl.ds(n * chunk, chunk)
        keyed = h // call.group * width
        views = stream_views(lambda i: (0, i), keys=call.given.views(k_ref, row, chunk, keyed, width),
                             **limb_views(call, prefix_refs, row, chunk, h * width, width),
                             **({"queries": call.given.views(q_ref, row, chunk, keyed, width)} if scored
                                else {}))
        (lefts, raised, separated), solve = staging[:3], staging[3:]
        # The own blocks' views, read into their tiles after the walk (`_read_out`).
        owned = [form.views(ref, row, chunk, h * algebra.SUB, algebra.SUB)
                 for form, ref in zip((placed, inverted), own_refs, strict=False)]

        def step(index, slots, carry, accumulators):
            del index
            slots.acquire_all()
            # Laid in the family's layout (`Family.tile_of`), as the scan lays it.
            whole = algebra.separated(*(family.tile_of(tile.refs[0]) for tile in limb_tiles(call, slots)))
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
                if j and len(raised) == 1:
                    # One buffer for every sub-chunk's keys (`_sub_chunk_staged`): the last sub-chunk's
                    # products read it before this one's keys overwrite it.
                    mosaic.synced(call.interpret)
                # A sub-chunk's keys staged and published once (`Family.staged_together`), the row sets' gated
                # rows with the first's, its products issued then.
                with family.staged_together():
                    keyed = operands.staged(family, gated_up(j + 1), raised[j % len(raised)]).T
                    held = held or [operands.staged(family, value, lefts[r]) for r, value in enumerate(gated)]
                for r, tile in enumerate(held):
                    at = _sum_index(call, r, j)
                    pair = ("pairs", tile.rows(first, algebra.SUB), keyed)
                    summed[at] = issuer.summed([pair], into=summed[at])
            return carry, tuple(summed)

        def read(carry, totals):
            del carry
            _read_out(call, products, totals, owned, solve, landing, sums, row=row, head=h,
                      growing=growing)

        return mosaic.Walk(views, step, steps=steps, sums=((algebra.SUB, chunk),) * (outputs * later),
                           read=read)

    blocks = pair_streams(call, keys, queries=scored)
    out_forms = [dataclasses.replace(call.held, rows=call.rows)] * outputs
    staging, landing = _sub_chunk_staged(call, keys, products=products)
    inputs = (q, k, *prefix, beta, taken, *own)
    return launch_kernel(program, call, out_type=tuple(form.shape(call.heads * chunk) for form in out_forms),
                         grid=(call.chunks, call.heads), grid_names=("chunks", "heads"), blocks=blocks,
                         stages=(1, 2), live=chunk_liveness(len(prefix) + 3), steps=steps,
                         model=_sub_chunk_flops(call, keys, outputs),
                         held=_sub_chunk_held(call, keys, outputs), staging=staging, landing=landing,
                         name=_named("recurrence_intra", products), sums=(chunk,) * (outputs * later),
                         swizzles=_pair_swizzles(call, blocks, keys))(*inputs)


def _read_out(call: Call, products: tuple[str, ...], totals, owned: list, solve: tuple, landing: dict,
              sums, *, row, head, growing: float) -> None:
    """The intra walk's outputs from its sums (`totals`, each a sub-chunk's row block `[SUB, C]` past the
    first's), a sub-chunk's rows at a time. The scores': the walk's (its first sub-chunk's none: no keys
    before it), scaled past one, beside each sub-chunk's own placed on its diagonal block (`placed`: the own
    blocks against an exact shifted identity, 'place'). The system's inverse, where the walk solves: its rows
    of blocks by forward substitution, `X_b = X_bb (E_b - L_b X)` for each sub-chunk `b` in turn, `X_bb` its
    own block's inverse (`own_pairs.intra_own`'s), `L_b` the walk's sums of its rows' pairs before its
    sub-chunk (`algebra.columns_before`: a row whose write gate overflows times a zero factor is a NaN past
    them), `X` the rows of blocks before it (staged, zero past them), `E_b` the identity's rows: fla_kda's
    merges, each stored less its unit diagonal, as `mainloop._weighted` takes it. The own blocks `owned`
    (their views, a form's parts each) are read into their tiles first, each thread its run of words
    (`own_loaded`)."""
    sub, chunk, family = algebra.SUB, call.chunk, call.family
    own_tiles, (place, lower, solved, stacked) = solve[0], solve[1:]
    held_once = {}
    for name, views, tiles, form in zip(("own_scores", "own_inverses"), owned, own_tiles,
                                        (placed_form(call), inverse_form(call)), strict=False):
        own_loaded(call, views, tiles)
        held_once[name] = operands.Parts(tiles, form.dtype)
    mosaic.synced(call.interpret)
    issuer = Issuer(call, {"place.exact": place, "lower": lower, "solve": solved})
    rows_of = jax.lax.broadcasted_iota(jnp.int32, (sub, chunk), 0)
    columns_of = jax.lax.broadcasted_iota(jnp.int32, (sub, chunk), 1)

    def identity(b):
        """The identity's rows of sub-chunk `b` `[SUB, C]`."""
        return jnp.where(columns_of == rows_of + b * sub, 1.0, 0.0)

    def rows_at(tile, b):
        return operands.Parts(tuple(ref.at[pl.ds(b * sub, sub)] for ref in tile.refs), tile.dtype)

    def stored(name, block, b):
        intra_block_stored(call, name, block, sums[0 if name == "scores" else 1], landing,
                           row=row + b * sub, head=head)

    for b in range(chunk // sub):
        placed = issuer.summed([("place", rows_at(held_once["own_scores"], b), product.Exact(identity(b)))])
        # The walk's sums before the sub-chunk alone (`algebra.columns_before`), as the own blocks sat on
        # their diagonal blocks once and the rest of the rows were zero, scaled past one.
        walked = (None if b == 0
                  else algebra.columns_before(totals[_sum_index(call, 0, b - 1)], b * sub) * growing)
        stored("scores", placed if walked is None else walked + placed, b)
    if "system" not in products:
        return
    inverted = inverse_form(call)
    # The rows of blocks staged zero, so a row block's sum over those not yet solved reads zero.
    for b in range(chunk // sub):
        zero = family.zeros((sub, chunk))
        operands.staged(family, zero, tuple(ref.at[pl.ds(b * sub, sub)] for ref in stacked))
    for b in range(chunk // sub):
        unit = identity(b)
        right = unit
        if b:
            pairs = algebra.columns_before(totals[_sum_index(call, 1, b - 1)], b * sub)
            right = unit - issuer.summed([("lower", pairs, operands.Parts(stacked, inverted.dtype))])
        found = issuer.summed([("solve", rows_at(held_once["own_inverses"], b), right)])
        stored("inverse", found - unit, b)
        if b + 1 < chunk // sub:
            family.settled(0)
            operands.staged(family, found, tuple(ref.at[pl.ds(b * sub, sub)] for ref in stacked))


def _pair_swizzles(call: Call, blocks: dict, keys: int) -> dict[str, int]:
    """The intra walk's streams' swizzles: each its widest at `keys` columns (`resources.widest_swizzle`), or
    unswizzled (`swizzling.UNSWIZZLED`) where an fp32 call with a decay a key dimension (its prefix's limbs,
    `Call.prefix_limbs`) takes mma.sync's products: Mosaic GPU synthesizes no shared-memory transfer from a
    128-byte swizzled fp32 tile into the register layout a TF32 product's operand takes beside that decay, so
    an A100's fast fp32 Kimi delta attention lowered no intra launch (row 12647); a decay a head's fp32 call
    lowers swizzled."""
    plain = call.family.name == "mma.sync" and call.given.stored == jnp.float32 and call.prefix_limbs > 0
    return {name: swizzling.UNSWIZZLED if plain else resources.widest_swizzle(dtype, keys)
            for name, (_, dtype) in blocks.items()}


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
    # After the walk (`_read_out`): a row block's placed scores, its sum and its identity beside the row and
    # column indices; solving, the substitution's sum, its right side and its rows of the inverse.
    read = [f32(algebra.SUB, chunk)] * (3 + 3 * (outputs > 1)) + masks(algebra.SUB, chunk) * 2
    return Held(carried=carried, phases=(
        [f32(chunk, block)] * (3 + outputs),
        [f32(chunk, block)] * 3 + masks(chunk, block) + held_tile(pairing, chunk, block) + paired,
        read))


def own_loaded(call: Call, views: tuple, tiles: tuple) -> None:
    """The own blocks' compact views (`own_pairs.intra_own`'s, a part each `[C, SUB]`) read into their plain
    `tiles`, each thread its run of words (`_own_copied`): a launch's tile read once, after its walk. The
    walk's products are mma.sync's (`sub_chunked`), whose operands load through the threads' own reads, so
    the warpgroup's barrier after the copy (`mosaic.synced`) is all their reads wait on. A load of the
    views into registers is no copy of them: a 16-column bf16 row is half of each 32-byte sector a warp's
    tiled load reads ("GMEM transfer does not access a whole number of complete 32 byte sectors", every
    Kimi Linear cell of the resource gate)."""
    for view, tile in zip(views, tiles, strict=True):
        if call.interpret:
            tile[...] = view[...]
            continue
        plgpu.inline_mgpu(arg_types=(plgpu.RefType(), plgpu.RefType()), return_type=())(
            lambda ctx, from_ref, to_ref: _own_copied(from_ref, to_ref))(view, tile)


def _own_copied(source_ref, tile_ref) -> None:
    """`own_loaded` in Mosaic GPU's terms: a thread's `C * SUB / THREADS` words, row by row."""
    index, i32 = ir.IndexType.get(), ir.IntegerType.get_signless(32)
    rows, columns = ir.MemRefType(source_ref.type).shape
    run = rows * columns // scan.THREADS

    def const(value):
        return arith.constant(i32, value)

    thread = arith.remui(utils.thread_idx(), const(scan.THREADS))
    for step in range(run):
        entry = arith.addi(arith.muli(thread, const(run)), const(step))
        at = [arith.index_cast(index, value)
              for value in (arith.divui(entry, const(columns)), arith.remui(entry, const(columns)))]
        memref.store(memref.load(source_ref, at), tile_ref, at)


def _sub_chunk_flops(call: Call, keys: int, outputs: int) -> int:
    """`_sub_chunk_walk`'s tensor FLOPs: each output's later sub-chunks' pairs a key block, the own scores
    placed (`[SUB, SUB] @ [SUB, C]` a sub-chunk), and solving, the substitution's sums over the rows of
    blocks (`[SUB, C] @ [C, C]`) and each own inverse's product (`[SUB, SUB] @ [SUB, C]`)."""
    del keys
    chunk, sub, count = call.chunk, algebra.SUB, call.chunk // algebra.SUB
    walked = outputs * 2 * _later(call) * sub * chunk * call.width
    placed = count * 2 * sub * sub * chunk
    solved = (_later(call) * 2 * sub * chunk * chunk + count * 2 * sub * sub * chunk) if outputs > 1 else 0
    return walked + placed + solved


def pair_streams(call: Call, keys: int, *, queries: bool) -> dict:
    """The walked streams of a launch forming a chunk's pairs at key block `keys` (`_sub_chunk_walk`'s,
    `own_pairs.intra_own`'s): the keys and the prefix a block at a time, and the queries where it forms the
    scores."""
    chunk = call.chunk
    return stream_blocks(keys=((chunk, keys), call.given), **limb_blocks(call, chunk, keys),
                         **({"queries": ((chunk, keys), call.given)} if queries else {}))


def _sub_chunk_staged(call: Call, keys: int, *, products: tuple[str, ...]) -> tuple:
    """`_sub_chunk_walk`'s staging at key block `keys`: each output's gated rows once, the later sub-chunks'
    keys gated up (`algebra.key_block_pairs`), and the step's separated prefix, plain (its rows read
    at the sub-chunks' first tokens); its landings a sub-chunk's row of pairs each output's
    (`intra_block_stored`). The keys gated up take one buffer every sub-chunk restages where the issuer's
    products are done when issued (mma.sync's, `Family.products_overlap` false), one a sub-chunk where its
    products may still read them as the next is staged: a buffer a sub-chunk held 32 KiB more a program at
    Kimi Linear's 64-key block in two parts."""
    chunk, pairing, sub = call.chunk, _pairing_operand(call), algebra.SUB
    held = dataclasses.replace(call.held, rows=call.rows)
    landing = {"square": held.landing(sub, chunk),
               **({"inverse": held.landing(sub, chunk)} if "system" in products else {})}
    lefts = tuple(staging_tiles(pairing, (chunk, keys))[0] for _ in products)
    raised = staging_tiles(pairing, *((chunk, keys),) * (_later(call) if call.family.products_overlap else 1))
    separated = swizzling.plain((2, chunk, keys), jnp.float32)
    # After the walk (`_read_out`): the placing identity's one exact part, in the placed scores' dtype; and
    # solving, the substitution's left and right operands and its rows of blocks, in the merges' form.
    placed, inverted = placed_form(call), inverse_form(call)
    solving = "system" in products
    # The own blocks' tiles (`own_loaded`), a 16-bit part each, plain: a thread's copy addresses them so.
    own = tuple(tuple(swizzling.plain((chunk, sub), form.stored) for _ in range(form.parts))
                for form in (placed, inverted)[:1 + solving])
    (place,) = staging_tiles(Operand(sub, 1, jnp.dtype(placed.stored)), (sub, chunk))
    solve = ((place, *staging_tiles(inverted, (sub, chunk), (sub, chunk), (chunk, chunk)))
             if solving else (place, (), (), ()))
    return (lefts, raised, separated, own, *solve), landing


def intra_rows(call: Call, name: str, operand: Callable[[str], jax.Array], beta) -> jax.Array:
    """An intra output's rows of a key block fp32: the scores' the queries; the system's each key scaled by
    its write strength (`beta`, the chunk's `[C]`), its pairs strict; each formed as its products are."""
    if name == "scores":
        return operand("queries")
    return algebra.write_gated(operand("keys"), call.family.rows_of(beta))


def pairing_parts(held: int) -> int:
    """The parts the factored pairs' fp32 operands take to their products (`algebra.key_block_pairs`) in a
    call holding its values in `held` parts: in a 16-bit call two bf16 parts, 16 of their bits, in the bytes
    one fp32 value stages in; an fp32 call's three. One TF32 part a pair, as fla's sub-chunk products take
    it, put Kimi delta attention's outputs under tiny gates at the floor at 8.3 times fla's distance from
    float64 at a sub-chunk's first tokens, where pairs across its edge cancel (M-JS-162 on a GH200's
    mma.sync, digit for digit an A100's at main, row 10382; the TF32 rounding interpreted gave the same
    digits, two parts 0.1 times fla's, p-kdavar)."""
    return product.bf16_part_count(jnp.float32, accurate=True) if held > 1 else 2


def _pairing_operand(call: Call) -> Operand:
    """The form the factored pairs' fp32 operands take (`pairing_parts`)."""
    return Operand(call.rows, pairing_parts(call.held.parts), jnp.dtype(jnp.float32))


