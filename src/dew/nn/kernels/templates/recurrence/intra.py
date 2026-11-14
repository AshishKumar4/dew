"""The recurrence template's intra launches under a decay a key dimension (Kimi delta attention's): each
chunk's pairs within it, the scores and the keys' system, walked a key block at a time on the mainloop's
skeleton (`mainloop.launch_kernel`), their factors formed once for both (the algebra's `key_block_pairs`),
and the system's inverse by doubling (`_intra_solve`), as fla_kda's intra rows and solve; each sub-chunk's
own pairs are `own_pairs`'. A seam of the mainloop of its own: these launches read its primitives, and
nothing of the mainloop reads them."""

import dataclasses
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import recurrence_algebra as algebra
from dew.nn.kernels.templates import mosaic, operands, product, resources, swizzling
from dew.nn.kernels.templates.cost import Work
from dew.nn.kernels.templates.operands import Issuer
from dew.nn.kernels.templates.recurrence.mainloop import (
    Call,
    Held,
    Operand,
    _doubling_held,
    _inverse_flops,
    _key_blocks,
    _merged,
    _merges,
    _merging,
    _resident,
    chunk_liveness,
    f32,
    issued_flops,
    key_block,
    launch_kernel,
    lowered_once,
    masks,
    product_fragments,
    staging_tiles,
    stream_blocks,
    stream_tile,
    stream_views,
)


def _intra_held(call: Call, block: int, outputs: int = 1) -> Held:
    """`_intra_walk`'s consumer thread at key block `block` forming `outputs` pairs' sums, which shared memory
    carries across the key blocks, one output's formed and added at a time: the block's prefix scan (its
    running pair, a shifted pair and the decay); then the factors every output shares beside the block's
    tiles: its prefix halves and their separated pair, the rows' gate down and the keys gated up
    (`algebra.key_block_pairs`), a pair's decay's two terms (`between`), a sub-chunk's middle factor and the
    laid references being rolled, an output's gated rows, their masks, one product's result `[C, right]`
    (`_pairs_right`) and its operands: its references read where they are needed and its products rolled, a
    product's columns put into a tile read whole once (`_placed`). The pairs read their rows' grid alone
    (`algebra.key_block_pairs`), one int32 tile: its columns' too stated 256 and 254 registers for an sm100's
    two intra launches, where ptxas held each at 248 spilling none, least 232 (armada
    recurrence-unused-grid-proof)."""
    chunk = call.chunk
    pairing = _pairing_operand(call)
    # A gated row block against the keys gated up, read transposed (`algebra.key_block_pairs`' 'pairs').
    paired = product_fragments(call, (pairing, chunk, block), (pairing, block, _pairs_right(call)),
                               rhs_transposed=True)
    if outputs == 1:
        # A lone output's sum carried in registers across the key blocks, as its launch held it before the
        # walks merged (M-JS-17's A100 parent), beside one gated output's factors.
        return Held(carried=[f32(chunk, chunk)], phases=(
            [f32(chunk, block)] * 5,
            [f32(chunk, block)] * 9 + [f32(chunk, _pairs_right(call))] + masks(chunk, block) + paired))
    # Live at once: the prefix halves and their separated pair, the rows' grid, the keys and the queries as
    # read, one output's rows and their gated form, the keys gated up, the gate down, a sub-chunk's middle
    # factor and the masked keys; the output's pairs and its sum read back. Its sums held in registers across
    # the key blocks, both outputs' at once, the merged launch spilled 336 to 424 bytes at 32 keys a block on
    # an A100 and an H100 (jsf-scan4 at d54ec6b02).
    return Held(carried=[], phases=(
        [f32(chunk, block)] * 5,
        [f32(chunk, block)] * 9 + [f32(chunk, _pairs_right(call))] + [f32(chunk, chunk)] * 2
        + masks(chunk, block) + paired))


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
                         stages=2, live=chunk_liveness(1), work=Work(flops=_inverse_flops(chunk), span=1),
                         hardware=issued_flops(call, _inverse_flops(chunk), *(_merged(call),) * 2),
                         held=Held(phases=(_doubling_held(call),)),
                         staging=_merges(call), landing={"square": held.landing(chunk, chunk)},
                         persistent=True)(
        system, taken)


@lowered_once
def _intra(q, k, g, beta, taken, call: Call, *, scale: float, solves: bool):
    """`(Aqk [R, HV * C], A [R, HV * C] where the call `solves`, hi, lo [R, HV * Dk])` of each chunk and value
    head under a decay a key dimension, one launch (`_intra_walk`): the scores as `call.held`, the keys'
    system where it `solves` (`systems`' form, the write-gated keys' pairs, strict), which `_intra_solve`
    inverts, and the decay's prefix halves (`call.gates`, `algebra.prefix`'s pair) from the laid decay `g`
    `[R, HV * Dk]` fp32, which every later launch reads. Each pair of sub-chunks' decay factored
    (`algebra.key_block_pairs`), their products on the tensor cores (fla_kda's intra rows), each sub-chunk's
    own pairs zero (a launch of their own, `own_pairs.intra_own`); the
    scores and the system share every factor in one walk, their sums in shared memory, one output's formed at
    a time (`_intra_walk`), where `_intra_merges` (wgmma's); else each output walks a launch of its own, its
    sum in registers."""
    products = ("scores", "system") if solves else ("scores",)
    if solves and not _intra_merges(call):
        scores, hi, lo = _intra_walk(q, k, g, beta, taken, call, scale=scale, products=("scores",))
        (system,) = _intra_walk(q, k, g, beta, taken, call, scale=scale, products=("system",))
        walked = (scores, system)
    else:
        *walked, hi, lo = _intra_walk(q, k, g, beta, taken, call, scale=scale, products=products)
    return (*walked, hi, lo)


def _intra_merges(call: Call) -> bool:
    """Whether the scores and the system take one intra walk (`_intra_walk`): where the family's products
    read their operands from shared memory (wgmma's), at a key block at which a multiprocessor holds a
    program of it, both sums in shared memory (an sm86's 99 KB held none of a narrow key dimension's). Where
    they read registers (mma.sync's), each output walks a launch of its own: one walk's sums through shared
    memory beside the fragments took 1.736 ms of an A100's Kimi Linear forward against the two launches'
    0.762 and 0.717 (M-JS-17, W33-a100-1), the forward 2.754 ms against 2.487."""
    if call.family.operands_in_registers:
        return False
    resident = functools.partial(_resident, call, functools.partial(_intra_held, call, outputs=2),
                                 functools.partial(_intra_staged, call, products=("scores", "system")),
                                 functools.partial(_intra_streamed, call, queries=True))
    return any(resident(block) for block in _key_blocks(call, contracted=True))


def _intra_prefix(call: Call, scanned, decays: jax.Array, keys: int) -> tuple:
    """A key block's prefix halves (`algebra.scanned_prefix`) of its `decays` `[C, keys]`, staged in
    `scanned` (each half's `C` zero rows, then its running pair, which the shifts read), left there at
    rows `C` on and returned in the accumulator's layout. The scan's steps run in a row-major layout,
    each thread's columns consecutive, whose transfers at any row offset touch every bank once: in the
    accumulator's, a fragment's 8 rows `keys` words apart shared 8 banks (19.0M bank conflicts on loads in a
    GH200's Kimi Linear intra launch, M-JS-17); the decay laid in and the prefix out once each."""
    chunk = call.chunk
    def half_at(half, rows):
        return scanned.at[half, rows]

    def stage(pair):
        # A thread reads other threads' rows: the last shift's reads done before the rows are written again,
        # and the writes made visible before they are read (`mosaic.synced`).
        mosaic.synced(call.interpret)
        for half, value in enumerate(pair):
            plgpu.store(half_at(half, pl.ds(chunk, chunk)), value, optimized=False)
        mosaic.synced(call.interpret)
        return lambda shift: tuple(_rows_of(half_at(half, pl.ds(chunk - shift, chunk)), keys)
                                   for half in (0, 1))

    mosaic.synced(call.interpret)
    for half in (0, 1):
        plgpu.store(half_at(half, pl.ds(0, chunk)), call.family.zeros((chunk, keys)), optimized=False)
    plgpu.store(half_at(0, pl.ds(chunk, chunk)), decays, optimized=False)
    mosaic.synced(call.interpret)
    stage(algebra.scanned_prefix(_rows_of(half_at(0, pl.ds(chunk, chunk)), keys), stage, chunk=chunk))
    return tuple(call.family.tile_of(half_at(half, pl.ds(chunk, chunk))) for half in (0, 1))


def _rows_of(ref, keys: int) -> jax.Array:
    """A `[C, keys]` tile `ref` of plain shared rows read in a row-major layout (`plgpu.Layout.WG_STRIDED`):
    each thread its consecutive words, a warp's transfer a contiguous run."""
    rows = ref.shape[0]
    layout = plgpu.Layout.WG_STRIDED((rows, keys), vec_size=min(4, rows * keys // 128))
    return plgpu.load(ref, layout=layout, optimized=False)


def _intra_summed(call: Call, found: jax.Array, at, *, first) -> None:
    """A key block's pairs `found` of an output added to its sum in shared memory `at` (written at the first
    block), before the next output's are formed (a merged walk's)."""
    mosaic.synced(call.interpret)

    # The first block's written, the rest added, in branches: a scalar condition laid over a tile
    # (`jnp.where`) is no broadcast Mosaic GPU's warpgroup semantics lower.
    @pl.when(first)
    def _():
        plgpu.store(at, found, optimized=False)

    @pl.when(jnp.logical_not(first))
    def _():
        plgpu.store(at, call.family.tile_of(at) + found, optimized=False)


def intra_sums_stored(call: Call, products: tuple[str, ...], totals: tuple, sums, landing: dict, *, row,
                       head, summed=None) -> None:
    """The intra walk's sums at its last step, each output's in its form: the scores as the held operand's,
    the system as `systems`' (fp32, as the sums are), through its landing, or a merged walk's own sum's slot
    of `summed` (`_intra_staged`), which holds it already."""
    chunk = call.chunk
    held = dataclasses.replace(call.held, rows=call.rows)
    for index, (name, total, ref) in enumerate(zip(products, totals, sums, strict=True)):
        if name == "scores":
            form, smem = held, landing["square"]
        else:
            form, smem = systems(call), landing["system"] if summed is None else summed.at[index]
        form.store(ref, total, row, chunk, head * chunk, chunk, family=call.family, smem=smem)


def _intra_walk(q, k, g, beta, taken, call: Call, *, scale: float, products: tuple[str, ...]):
    """The intra launch's walk of each chunk's key blocks (`key_block`, the block a multiprocessor holds most
    programs of): each step the block's prefix halves from its decay (`algebra.scanned_prefix`, staged in a
    padded shared tile whose rows the sub-chunks' references read), stored for the later launches, and the
    block's pairs summed into the sums the walk carries, stored at its last step: the scores (the queries'
    pairs, scaled) and, under the delta rule, the keys' system (the write-gated keys' pairs, strict), their
    factors formed once for both (`algebra.key_block_pairs`), and each sub-chunk's own pairs each their own
    decay (`algebra.key_diagonal_pairs`).
    Its prefix an XLA scan of its own, thirteen launches took 0.30 ms of Kimi Linear's 1.57 ms forward on an
    H100; the scores and the system a launch each, each scanned the prefix and gated its keys again, 0.42
    and 0.40 ms of its 1.45 (kda-batch-h100-7e3a5fb48)."""
    chunk = call.chunk
    width, gates, family = call.width, call.gates, call.family
    held = dataclasses.replace(call.held, rows=call.rows)
    # The scores' launch stores the prefix halves; a launch of the system alone reads no queries.
    scored = "scores" in products
    outputs = len(products)
    keys = key_block(call, functools.partial(_intra_held, call, outputs=outputs),
                     functools.partial(_intra_staged, call, products=products),
                     functools.partial(_intra_streamed, call, queries=scored), contracted=True)
    steps = width // keys

    def program(at, q_ref, k_ref, g_ref, beta_ref, taken_ref, *outs, streams, staging, landing):
        n, h = at["chunks"], at["heads"]
        row, span = n * chunk, pl.ds(n * chunk, chunk)
        keyed = h // call.group * width
        views = stream_views(lambda i: (0, i), keys=call.given.views(k_ref, row, chunk, keyed, width),
                             decays=gates.views(g_ref, row, chunk, h * width, width),
                             **({"queries": call.given.views(q_ref, row, chunk, keyed, width)} if scored
                                else {}))
        pairing, scanned, placing, *rest = staging
        summed_at = rest[0] if outputs > 1 else None
        sums, prefixes = outs[:outputs], outs[outputs:]

        def step(index, slots, totals, accumulators):
            slots.acquire_all()
            prefix = _intra_prefix(call, scanned, stream_tile(slots, "decays", gates).value(), keys)
            for half, ref in zip(prefix if scored else (), prefixes, strict=True):
                gates.store(ref, half, row, chunk, h * width + index * keys, keys, family=family,
                            smem=landing["prefix"])

            def at_token(token):
                """The prefix halves at the chunk's `token`, a column's vectors, from the staged scan."""
                return tuple(family.columns_of(scanned.at[half, chunk + token]) for half in (0, 1))

            def operand(name):
                return stream_tile(slots, name, call.given).value().astype(jnp.float32)

            def rows(name):
                return intra_rows(call, name, operand, beta_ref.at[h, span])

            keyed, *gated = pairing
            pairs = algebra.key_block_pairs([functools.partial(rows, name) for name in products],
                                            operand("keys"), prefix, at_token,
                                            Issuer(call, {"pairs": gated}).summed,
                                            lambda up: operands.staged(family, up.astype(jnp.float32),
                                                                       keyed),
                                            _placed(call, placing), chunk=chunk, right=_pairs_right(call))
            # A key block's pairs are zero on and above the sub-chunks' diagonal blocks as formed
            # (`algebra.key_block_pairs`: rows of a sub-chunk against an earlier one's keys alone), so they
            # are summed as they come, the scores scaled each block (unscaled, two blocks' sums near fp32's
            # largest overflow) and the system masked once at the last step (a row whose write gate
            # overflows times a zero factor is a NaN there, which its strict mask clears).
            formed = []
            for name, found in zip(products, pairs, strict=True):
                formed.append(found * scale if name == "scores" else found)
                if summed_at is not None:
                    _intra_summed(call, formed[-1], summed_at.at[len(formed) - 1], first=index == 0)
            if summed_at is None:
                totals = (totals[0] + formed[0],)

            @pl.when(index == steps - 1)
            def _():
                if summed_at is not None:
                    mosaic.synced(call.interpret)
                found = (totals if summed_at is None
                         else tuple(family.tile_of(summed_at.at[o]) for o in range(outputs)))
                masked = tuple(algebra.strictly_lower(total) if name == "system" else total
                               for name, total in zip(products, found, strict=True))
                intra_sums_stored(call, products, masked, sums, landing, row=row, head=h, summed=summed_at)

            return totals, accumulators

        carry = () if outputs > 1 else (family.as_tile(family.zeros((chunk, chunk))),)
        return mosaic.Walk(views, step, steps=steps, carry=carry)

    blocks = _intra_streamed(call, keys, queries=scored)
    formed = [held if name == "scores" else systems(call) for name in products]
    out_type = (*(form.shape(call.heads * chunk) for form in formed),
                *((gates.shape(call.heads * width),) * 2 if scored else ()))
    # Each output's product a key sub-chunk's columns; the exponentials a key block's rows gated down and its
    # keys up.
    work = Work(flops=outputs * 2 * chunk * chunk * width, span=steps,
                exponentials=2 * chunk * width)
    staging, landing = _intra_staged(call, keys, products=products)
    # Its rings as deep as its residency holds them (`resources.plan`): one deep, a B200's Kimi Linear walk
    # waited on each key block's loads at two programs a multiprocessor, 24.5% of its warps active and
    # long_scoreboard 6.2 cycles an issued instruction, 819 us against fla's intra rows' 589 (M-JS-118).
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=blocks, stages=(1, 2),
                         live=chunk_liveness(4), work=work,
                         hardware=issued_flops(call, work.flops, *(_pairing_operand(call),) * 2),
                         held=_intra_held(call, keys, outputs), staging=staging, landing=landing,
                         name="recurrence_intra" if scored else "recurrence_intra_system",
                         # Each stream staged at its key block's row: a tile slices only whole swizzle rows.
                         swizzles={name: resources.widest_swizzle(dtype, keys)
                                   for name, (_, dtype) in blocks.items()})(q, k, g, beta, taken)


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
    """`_intra_walk`'s staging tiles and landings at key block `keys` forming `products`: the pairs' operands
    (`_pairing`), the prefix scan's halves below as many zero rows (`algebra.scanned_prefix`'s shifts), each
    output's sum across the key blocks, and the stores' landings: the
    scores' square and a block of the prefix halves where it forms the scores, the system's where it forms the
    system alone; a merged walk's system lands through its own sum's slot, fp32 as the system is
    (`intra_sums_stored`): its own 16 KB tile held a B200's merged walk 2.6 KB past two programs' share
    (118,844 bytes at a 16-column key block)."""
    chunk = call.chunk
    merged = len(products) > 1
    landing = {**({"square": dataclasses.replace(call.held, rows=call.rows).landing(chunk, chunk),
                   "prefix": call.gates.landing(chunk, keys)} if "scores" in products else {}),
               **({"system": systems(call).landing(chunk, chunk)}
                  if "system" in products and not merged else {})}
    # Read at row offsets no tiling divides (the scan's shifts, the sub-chunks' references), so plain.
    # Unpadded: a tile read in the accumulator's layout takes a row stride a whole number of its tiling's
    # columns (Mosaic GPU's layout inference refused 24 at a 16-column tiling and 80 at a 64-column one,
    # jsf-kdaintra5 and 9), a multiple of 32 words at 32 columns or more, so no stride it takes moves a
    # fragment store's rows off one another's banks.
    scanned = swizzling.plain((2, 2 * chunk, keys), jnp.float32)
    # A merged walk's sums in shared memory, each read and written whole, so in swizzled 8-row tiles: plain,
    # rows a multiple of 32 words apart put a fragment's 8 rows in the same banks (41.3M bank conflicts in a
    # GH200's Kimi Linear intra launch at ceb64ccc5d, M-JS-17); a lone output's in registers (`_intra_held`).
    sums = (swizzling.staged((len(products), chunk, chunk), jnp.float32, 128),) if merged else ()
    # Each product's columns put into a tile read whole once (`algebra.key_block_pairs`' `placed`).
    placing = swizzling.plain((chunk, chunk), jnp.float32)
    return (_pairing(call, keys), scanned, placing, *sums), landing


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


def _pairs_right(call: Call) -> int:
    """The keys a factored pairs' product takes (`algebra.key_block_pairs`' `right`): a sub-chunk's, or the
    narrowest columns the family's product takes where wider (tcgen05's 32: a 64-row product splits its N
    columns between two lane groups)."""
    return max(algebra.SUB, call.family.narrowest_columns(jnp.bfloat16))


def _placed(call: Call, tile) -> tuple:
    """`algebra.key_block_pairs`' `placed` on the plain tile `tile` `[C, C]` fp32: a sub-chunk `t`'s columns
    put at `t * SUB`, and the whole read once, its writes waited on before (`mosaic.synced`) and the reads
    before the next output's writes."""
    def put(t, value):
        plgpu.store(tile.at[:, pl.ds(t * algebra.SUB, algebra.SUB)], value, optimized=False)

    def read():
        mosaic.synced(call.interpret)
        value = call.family.tile_of(tile)
        mosaic.synced(call.interpret)
        return value

    return put, read


def _pairing(call: Call, block: int) -> list:
    """The factored pairs' staging: the keys gated up, staged once for every output's products
    (`algebra.key_block_pairs`' `staged`), then the gated rows' two sets (`Issuer`'s 'pairs', one formed
    while the other's product reads it), each a `[C, block]` tile a part (`_pairing_operand`)."""
    return [staging_tiles(_pairing_operand(call), (call.chunk, block))[0] for _ in range(3)]
