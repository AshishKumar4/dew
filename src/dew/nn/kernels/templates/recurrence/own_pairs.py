"""Each sub-chunk's own pairs under a decay a key dimension, launches of their own beside the factored walks
(`mainloop._intra_walk`, `key_backward._key_pairs_back`): the forward's bands (`algebra.key_diagonal_bands`)
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

from dew.nn.kernels import recurrence_algebra as algebra
from dew.nn.kernels.templates import mosaic, resources
from dew.nn.kernels.templates.recurrence.mainloop import (
    Call,
    Held,
    Work,
    chunk_liveness,
    f32,
    fp32_operand,
    held_tile,
    intra_masked,
    intra_rows,
    intra_sums_stored,
    key_block,
    launch_kernel,
    masks,
    stream_blocks,
    stream_tile,
    stream_views,
    systems,
)


def shifted_keys(call: Call, keys: jax.Array, prefix: tuple):
    """A key block's keys and separated prefix (`algebra.separated` of its halves `prefix`), each offset's
    reads them shifted by it (`algebra.Shifted`) in registers (`Issuer.row_shifted`, a sub-chunk a group:
    the pairs a sub-chunk's own), any value where a row's source precedes its sub-chunk (masked). Staged in
    shared memory and read shifted, an A100's own pairs lost 74% of their loads' wavefronts to bank
    conflicts (M-JS-62, row 5305)."""
    held = (keys, *algebra.separated(*prefix))
    return lambda e: tuple(call.family.row_shifted(value, e, period=algebra.SUB, interpret=call.interpret)
                           for value in held)


def shifted_up(call: Call, e, value: jax.Array) -> jax.Array:
    """`value`'s row `i` at row `i - e` (`algebra.key_diagonal_pairs_back`'s `lower`), in registers as
    `shifted_keys`' are, any value where a row's source leaves its sub-chunk (masked)."""
    return call.family.row_shifted(value, -e, period=algebra.SUB, interpret=call.interpret)


def intra_own(q, k, hi, lo, beta, taken, call: Call, *, scale: float, products: tuple[str, ...]) -> tuple:
    """Each output of the intra walk's (`products`) sub-chunks' own pairs, each its own decay, `[R, HV * C]`
    in the output's form, zero but on the sub-chunks' diagonal blocks, which the walk's sums leave zero
    (`algebra.key_block_pairs` takes the keys before a sub-chunk alone), so each entry is rounded once and
    their sum is exact (`chunk_forward.prepared`): a launch of their own, no products on the tensor cores, as
    fla_kda's token-parallel diagonal is (chunk_kda_fwd_intra_token_parallel). It walks each chunk's key
    blocks, the keys, queries and the walk's stored prefix halves `hi`, `lo` a block at a time, summing each
    output's bands (`algebra.key_diagonal_bands`) and placing them once at its last step (`algebra.banded`).
    In the walk, they held an H100's Kimi Linear intra launch at 248 registers, 1982 us against 492
    (M-JS-50d)."""
    chunk, width, gates, family = call.chunk, call.width, call.gates, call.family
    scored = "scores" in products
    keys = key_block(call, functools.partial(_own_held, call, outputs=len(products)),
                     functools.partial(_own_staged, call, products=products),
                     functools.partial(_own_streamed, call, queries=scored))
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
        del staging

        def step(index, slots, bands, accumulators):
            slots.acquire_all()

            def operand(name):
                return stream_tile(slots, name, call.given).value().astype(jnp.float32)

            prefix = tuple(stream_tile(slots, name, gates).value() for name in ("highs", "lows"))
            found = algebra.key_diagonal_bands(
                [intra_rows(call, name, operand, beta_ref.at[h, span]) for name in products],
                shifted_keys(call, operand("keys"), prefix), algebra.separated(*prefix))
            bands = tuple(band + more for band, more in zip(bands, found, strict=True))

            @pl.when(index == steps - 1)
            def _():
                intra_sums_stored(call, products, tuple(
                    intra_masked(algebra.banded(band), strict=name == "system", scale=scale)
                    for name, band in zip(products, bands, strict=True)), sums, landing, row=row, head=h)

            return bands, accumulators

        carry = tuple(family.as_tile(family.zeros((chunk, algebra.SUB))) for _ in products)
        return mosaic.Walk(views, step, steps=steps, carry=carry)

    blocks = _own_streamed(call, keys, queries=scored)
    out_type = tuple((call.held if name == "scores" else systems(call)).shape(call.heads * chunk)
                     for name in products)
    # Each output's pairs a key a term, a sub-chunk's offsets; the decay's exponentials once an offset.
    work = Work(flops=len(products) * 3 * chunk * algebra.SUB * width, span=steps,
                exponentials=chunk * algebra.SUB * width)
    staging, landing = _own_staged(call, keys, products=products)
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=blocks, stages=(1, 2),
                         live=chunk_liveness(5), work=work, held=_own_held(call, keys, len(products)),
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


def _own_staged(call: Call, keys: int, *, products: tuple[str, ...]) -> tuple:
    """`intra_own`'s staging at key block `keys`: none (its shifted reads in registers, `shifted_keys`), and
    each output's store's landing."""
    del keys
    chunk = call.chunk
    landing = {**({"square": dataclasses.replace(call.held, rows=call.rows).landing(chunk, chunk)}
                  if "scores" in products else {}),
               **({"system": systems(call).landing(chunk, chunk)} if "system" in products else {})}
    return (), landing


def _own_held(call: Call, block: int, outputs: int = 1) -> Held:
    """`intra_own`'s consumer thread at key block `block` forming `outputs` outputs' bands, carried across
    the key blocks: an offset's rows, keys and separated prefix as read, the row's prefix, the decay and a
    term, beside the keys and separated prefix held across the offsets (`shifted_keys`, `Issuer.row_shifted`:
    ptxas's least spill-free cap at a Kimi Linear call's blocks 200 on an A100 and 184 on an H100 at 3
    tiles more, 248 and 232 at none, jsf-ownneed); at the last step one output's pairs placed `[C, C]` and
    the placing grids."""
    chunk = call.chunk
    return Held(carried=[f32(chunk, algebra.SUB)] * outputs, phases=(
        [f32(chunk, block)] * (outputs + 11),
        [f32(chunk, chunk)] * 4))


def pairs_back_views(call: Call, row: int, h, q_ref, k_ref, hi_ref, lo_ref, ds_ref, da_ref) -> dict:
    """The streams of a key pairs backward's program at chunk `row` and value head `h` (`key_pairs_own`'s,
    `key_backward._key_pairs_back`'s): the pairs' gradients a chunk's square once, and the queries, keys and
    the separated prefix's halves a key block a step."""
    chunk, width = call.chunk, call.width
    given, gates, square = call.given, call.gates, fp32_operand(call)
    keyed = h // call.group * width
    return {**stream_views(lambda i: (0, 0), scores=square.views(ds_ref, row, chunk, h * chunk, chunk),
                           system=square.views(da_ref, row, chunk, h * chunk, chunk)),
            **stream_views(lambda i: (0, i),
                           queries=given.views(q_ref, row, chunk, keyed, width),
                           keys=given.views(k_ref, row, chunk, keyed, width),
                           gates=gates.views(hi_ref, row, chunk, h * width, width),
                           lows=gates.views(lo_ref, row, chunk, h * width, width))}


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
    gradients `d_scores` and `d_system`: a launch of their own, no products on the tensor cores, as the
    forward's (`intra_own`), each key block's offsets reading the pairs' gradients afresh and the
    keys and separated prefix shifted (`shifted_keys`). In `_key_pairs_back`'s walk, they held an
    H100's Kimi Linear launch at 248 registers, 2268 us against 672 (M-JS-50d)."""
    chunk, width, family = call.chunk, call.width, call.family
    given, gates, square = call.given, call.gates, fp32_operand(call)
    # The block a multiprocessor holds most programs of, at its registers and its shared memory: by its
    # registers alone, an sm86's 99 KB held none of its 32-key block's 106 KB (row 5282).
    keys = key_block(call, functools.partial(_key_own_held, call), functools.partial(_key_own_staged, call),
                     functools.partial(_key_own_streamed, call))
    steps = width // keys

    def program(at, q_ref, k_ref, hi_ref, lo_ref, beta_ref, ds_ref, da_ref, taken_ref, dq_ref, dk_ref, dg_ref,
                db_ref, *, streams, staging, landing):
        del taken_ref
        del staging
        n, h = at["chunks"], at["heads"]
        row, span = n * chunk, pl.ds(n * chunk, chunk)
        views = pairs_back_views(call, row, h, q_ref, k_ref, hi_ref, lo_ref, ds_ref, da_ref)

        def step(index, slots, d_betas, accumulators):
            slots.acquire_all()
            betas = family.rows_of(beta_ref.at[h, span])
            query = stream_tile(slots, "queries", given).value().astype(jnp.float32)
            key = stream_tile(slots, "keys", given).value().astype(jnp.float32)
            prefix = stream_tile(slots, "gates", gates).value(), stream_tile(slots, "lows", gates).value()
            # Each pairs' gradient read afresh at each offset, the scores' scaled as their pairs were.
            def loaded(name, scaled):
                return stream_tile(slots, name, square).value() * scaled

            d_loads = [functools.partial(loaded, name, scaled)
                       for name, scaled in (("scores", scale), ("system", 1.0))]
            zero = jnp.zeros(key.shape, jnp.float32)
            def rows(name):
                return stream_tile(slots, name, given).value().astype(jnp.float32)

            (d_query, d_keyed), d_key = algebra.key_diagonal_pairs_back(
                d_loads, [functools.partial(rows, "queries"), lambda: rows("keys") * betas[:, None]],
                algebra.separated(*prefix),
                shifted_keys(call, key, prefix), functools.partial(shifted_up, call),
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
    # Each offset's two pairs' terms a key, forward and moved up; the decay's exponentials once an offset.
    terms = 2 * 2 * 3 * chunk * algebra.SUB * width
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=chunk_liveness(7),
                         work=Work(flops=terms, span=steps, exponentials=algebra.SUB * chunk * width),
                         held=_key_own_held(call, keys), staging=staging, landing=landing,
                         name="recurrence_key_pairs_own", issues=False)(q, k, hi, lo, beta, d_scores,
                                                                       d_system, taken)


def _key_own_streamed(call: Call, keys: int) -> dict:
    """`key_pairs_own`'s streams at key block `keys`: the pairs' gradients once, the queries, keys and prefix
    halves a block at a time."""
    chunk, given, gates, square = call.chunk, call.given, call.gates, fp32_operand(call)
    return stream_blocks(scores=((chunk, chunk), square), system=((chunk, chunk), square),
                         queries=((chunk, keys), given), keys=((chunk, keys), given),
                         gates=((chunk, keys), gates), lows=((chunk, keys), gates))


def _key_own_staged(call: Call, keys: int) -> tuple:
    """`key_pairs_own`'s staging at key block `keys`: none (its shifted reads in registers, `shifted_keys`,
    `shifted_up`); a key block's landing."""
    return (), {"block": fp32_operand(call).landing(call.chunk, keys)}


def _key_own_held(call: Call, keys: int) -> Held:
    """`key_pairs_own`'s consumer thread at key block `keys`, as an sm100's lowered peak counts it (row 5298,
    jsf-sm100-backward): the write gates' terms carried, a row's vector; at an offset's shifted loads the
    three gradients carried, the separated prefix's two tiles and the three shifted reads (`shifted_keys`:
    their sources held across the offsets in the registers the shared memory's loads took, ptxas's least
    spill-free cap at a Kimi Linear call's block within this count, 224 on a B200, 160 on an H100,
    jsf-ownneed), the queries and
    keys as read, the square bands' coordinates and masks and a row's vector; at its bands' reads seven of
    those tiles, a pairs' gradient `[C, C]` where products read shared memory (beside them in registers
    where they read registers), the queries and keys and three row vectors."""
    chunk = call.chunk
    tile, given = f32(chunk, keys), held_tile(call.given, chunk, keys)
    row = f32(chunk, 4)
    banded = masks(chunk, chunk) + masks(chunk, chunk // 2) + masks(chunk, keys)
    square = [] if call.family.operands_in_registers else [f32(chunk, chunk)]
    return Held(carried=[row], phases=([tile] * 8 + given * 2 + banded + [row],
                                       [tile] * 7 + square + given * 2 + banded + [row] * 3))
