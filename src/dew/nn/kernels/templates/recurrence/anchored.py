"""Each sub-chunk's own pairs under a decay a key dimension as anchored products on the tensor cores, where
the call's layer bounds its decays a token (`anchoring`, Kimi delta attention's): fla_kda's safe-gate diagonal
(chunk_intra.py at 413ad2c), each pair's decay `exp2(g_i - g_j)` split at its sub-chunk's middle token `A`
into a factor on its row, `exp2(g_i - A)`, and one on its key, `exp2(A - g_j)` (`algebra.own_anchored`), so a
key block's own pairs (`own_pairs.intra_own`'s, the forward's) are one product of scaled rows and keys, each
in two bf16 parts (`intra._pairing_operand`: one part cancelled a packed document's first token, row 5194).
A key block takes its pairs each its own decay (the exact path) wherever its guard fails: a factor past
`algebra.OWN_REACH`, or a scaled operand's element past `algebra.OWN_BAND` or flushed where it was given
nonzero. The backward's own pairs (`own_pairs.key_pairs_own`) stay the exact path's: anchored products beside
it in one launch took its registers past ptxas's 255 on one family or another under every form tried."""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import recurrence_algebra as algebra
from dew.nn.kernels.templates import mosaic, operands
from dew.nn.kernels.templates.operands import Issuer
from dew.nn.kernels.templates.recurrence.intra import _pairing_operand
from dew.nn.kernels.templates.recurrence.mainloop import Call, Operand, f32, fp32_operand, staging_tiles


def anchoring(call: Call, *, reading: bool = False) -> bool:
    """Whether a call's forward own pairs are anchored products a key block (`step`): its decay a key
    dimension and its layer bounds it a token (`Call.decay_bound`: Kimi delta attention's -5, a sub-chunk's
    factors within 2^58 from its middle, its guard holding on all but steep draws), its system's rows its
    keys (no erase along reads), and its operands 16-bit, as fla_kda's safe-gate diagonal takes them: an
    fp32 call's queries and keys, its own 24 bits, staged in two bf16 parts lost a score cancelling below
    them (`(1 + 2^-9 + 2^-20, 1 + 2^-9)` against `(1/2, -1/2)` to zero, its exact `2^-21`, the head
    review's P0 of 6496f845f), which its pairs each their own decay keep."""
    return (call.per_key and not reading and call.decay_bound is not None
            and jnp.dtype(call.given.dtype).itemsize == 2)


def prefix_at(call: Call, prefix_refs: tuple, row, column, keys: int, token) -> tuple:
    """The key block's separated prefix (`algebra.separated`) at the chunk's `token`, a column's vectors,
    read from its stored limbs (`intra._prefix`'s) at the chunk's `row` and the block's `column`."""
    return algebra.separated(*(call.family.columns_of(ref.at[row + token, pl.ds(column, keys)])
                               for ref in prefix_refs))


def forward_staging(call: Call, keys: int, outputs: int) -> tuple:
    """`step`'s staging at key block `keys`: the scaled keys' two parts, then each output's scaled rows'."""
    return staging_tiles(_pairing_operand(call), *((call.chunk, keys),) * (1 + outputs))


def forward_held(chunk: int, block: int, outputs: int) -> tuple:
    """`step`'s phases beside the exact path's at key block `block`: its guard's (the separated prefix and
    the anchors' halves, the factors, the keys, an output's rows as given and scaled as they are staged) and
    its products' (each output's `[C, C]` sum and the compact blocks taken from it beside what the exact
    path holds across its offsets): sized as its first form's, whose guard held every row and the keys
    together, green at every target's gate (d93018eb2)."""
    tile = f32(chunk, block)
    return [tile] * (outputs + 6), [tile] * (outputs + 5) + [f32(chunk, chunk), f32(chunk, algebra.SUB)]


def _staged(call: Call, given: jax.Array, scaled: jax.Array, tiles, flags: jax.Array):
    """`scaled` staged as its products' parts in `tiles` (`operands.staged`), and the guard's row flags
    `flags` with its own (`algebra.own_rows_failed` of `given` and it): flagged and staged as it is formed, so
    no scaled operand is held beside another (an A100's and an H100's launches spilled holding them all)."""
    return (operands.staged(call, scaled, tiles),
            jnp.maximum(flags, algebra.own_rows_failed(None, [(given, scaled)])))


def step(call: Call, rows: Sequence[Callable[[], jax.Array]], key: jax.Array, limbs: tuple,
         at_token: Callable, staged: tuple, bands, exact: Callable, summed=None):
    """A key block's own pairs of each output's rows `[C, keys]` (`rows`, each formed as it is scaled and
    staged, so no output's is held beside another's) against its `key`s, one anchored product
    each (`algebra.own_anchored` at each sub-chunk's middle token), added into the bands (`bands`, or natively
    `summed`, `own_pairs.own_bands`' tile), staged in `staged` (`forward_staging`); else, where its guard
    fails, `exact(bands)`, the pairs each its own decay. Every term of a pair it keeps is its pair's own
    (its factors' product its decay), so only an operand's own scaling can lose one: its guard holds each
    scaled element in band (`algebra.own_out_of_band`)."""
    whole = algebra.separated(*limbs)
    anchors = algebra.own_anchors(at_token, key.shape)
    flags = algebra.own_rows_failed(algebra.own_reach_exceeded(whole, anchors))
    down, up = algebra.own_anchored(whole, anchors)
    keys_staged, flags = _staged(call, key, key * up, staged[0], flags)
    rows_staged = []
    for formed, tiles in zip(rows, staged[1:], strict=True):
        value = formed()
        left, flags = _staged(call, value, value * down, tiles, flags)
        rows_staged.append(left)

    def anchored(bands):
        issuer = Issuer(call)
        found = [algebra.own_compact(issuer.summed([("own", left, keys_staged.T)])) for left in rows_staged]
        if summed is None:
            return tuple(band + more for band, more in zip(bands, found, strict=True))
        for o, more in enumerate(found):
            plgpu.store(summed.at[o], call.family.tile_of(summed.at[o]) + more, optimized=False)
        mosaic.synced(call.interpret)
        return bands

    return lax.cond(jnp.max(flags) > 0.0, exact, anchored, bands)


BLOCK_PARTS = 3
"""The bf16 parts the backward's pairs' gradients' own blocks take to its products: 24 bits, an fp32 value's
own, so no cotangent loses a bit the exact path's fp32 terms keep (`product.terms`' five pairs against the
scaled rows' and keys' two). In two (16 bits) a score cotangent of `1 + 2^-9 + 2^-20` staged as another of
`1 + 2^-9`, its query's gradient of their difference zero where the exact path's is `2^-20` (a head review's
P0 of 28c4e9f0f)."""

COLUMNS = 4
"""The compact blocks `back` reads (`algebra.own_columns`): the scores' and its transposed's, the system's
and its transposed's."""


def back_staging(call: Call, keys: int) -> tuple:
    """`back`'s staging at key block `keys`: the scaled keys', queries' and gated keys' two parts, then a
    sub-chunk's compact block's `BLOCK_PARTS`, each sub-chunk's in turn."""
    chunk = call.chunk
    block = Operand(call.rows, BLOCK_PARTS, jnp.dtype(jnp.float32))
    return (*staging_tiles(_pairing_operand(call), *((chunk, keys),) * 3),
            *staging_tiles(block, (chunk, algebra.SUB)))


def back_held(chunk: int, keys: int, banded: list) -> tuple:
    """`back`'s phases beside the exact path's at key block `keys`: its guard's (a pairs' gradients' block as
    it is read, the separated prefix and an anchor's halves, its factor, an operand as given and scaled, the
    masks) and its products' (one gradient's sum, the compact blocks it reads and a sub-chunk's of them with
    its parts, then the outer factors: the separated prefix, an anchor's halves and a factor beside the
    three gradients): at 32 keys an RTX A6000's and an RTX 4090's took 245 of its 234."""
    tile, compact = f32(chunk, keys), f32(chunk, algebra.SUB)
    return [tile] * 10 + [compact] * 2 + banded, [tile] * 7 + [compact] * (2 + 2 * BLOCK_PARTS)


def back_anchoring(call: Call, *, reading: bool = False) -> bool:
    """Whether a call's backward own pairs are anchored products a key block (`back`), as its forward's are
    (`anchoring`)."""
    return anchoring(call, reading=reading)


def back(call: Call, columns: tuple, at: tuple, scale: float, rows: tuple[Callable, Callable, Callable],
         limbs: Callable, at_token: Callable, staged: tuple, exact: Callable) -> tuple:
    """`algebra.key_diagonal_pairs_back`'s `((d_query, d_keyed), d_key)` of a key block, its own pairs
    anchored products where its guard holds, else `exact()`: the pairs' gradients' own blocks read compact
    (`columns`, the scores' and the system's `algebra.own_columns` and their transposed's `[R, HV * SUB]`, at
    `at`: the chunk's row and the head's column), the scores' scaled as their strips are; of `rows` (the
    queries', the gated keys' and the keys' reads, each read where it is taken) and the prefix's limbs'
    read (`limbs`), the rows' gradients `exp2(g_i - F) (dP K~)` with `K~ = K exp2(F - g_j)` against each
    sub-chunk's first token `F`, and the keys' `exp2(L - g_j) (dS^T Q~ + dA^T R~)` with the rows scaled by
    `exp2(g_i - L)` against its last `L`: each outer factor at most one, so a term below fp32's range there
    is below it in its pair, and a term past it its guard's (`algebra.OWN_PRODUCT`, the largest block's
    element times the largest scaled operand's). Each gradient summed in a scope of its own, a sub-chunk at
    a time (`_by_sub_chunk`), the rows' gradients scaled before the keys' is summed. Each scaled operand in
    two bf16 parts, staged in `staged` (`back_staging`)."""
    chunk = call.chunk
    query, gated, key = rows
    read = functools.partial(_columns, call, at)
    flags, largest = jnp.zeros((chunk,), jnp.float32), jnp.zeros((chunk,), jnp.float32)
    for ref, factor in zip(columns[::2], (scale, 1.0), strict=True):
        found = read(ref, factor)
        flags = jnp.maximum(flags, algebra.own_rows_failed(None, [(found, found)]))
        largest = jnp.maximum(largest, jnp.max(jnp.abs(found), axis=1))
    # Each scaled operand flagged and let go: the branch forms and stages them again.
    whole = algebra.separated(*limbs())
    keys = key()
    shape = keys.shape
    first = algebra.own_anchors(at_token, shape, algebra.FIRST)
    keyed = keys * algebra.between(first, whole)
    flags = jnp.maximum(flags, algebra.own_rows_failed(algebra.own_reach_exceeded(whole, first),
                                                       [(keys, keyed)]))
    reach = jnp.max(jnp.abs(keyed), axis=1)
    last = algebra.own_anchors(at_token, shape, algebra.LAST)
    flags = jnp.maximum(flags, algebra.own_rows_failed(algebra.own_reach_exceeded(whole, last)))
    for formed in (query, gated):
        value = formed()
        scaled = value * algebra.between(whole, last)
        reach = jnp.maximum(reach, jnp.max(jnp.abs(scaled), axis=1))
        flags = jnp.maximum(flags, algebra.own_rows_failed(None, [(value, scaled)]))
    failed = (jnp.max(flags) > 0.0) | (jnp.max(largest) * jnp.max(reach) > 2.0 ** algebra.OWN_PRODUCT)

    def anchored():
        separated = algebra.separated(*limbs())
        first = algebra.own_anchors(at_token, shape, algebra.FIRST)
        keys_staged = operands.staged(call, key() * algebra.between(first, separated), staged[0])
        inner = algebra.between(separated, algebra.own_anchors(at_token, shape, algebra.LAST))
        queries_staged = operands.staged(call, query() * inner, staged[1])
        gated_staged = operands.staged(call, gated() * inner, staged[2])
        summed = functools.partial(_by_sub_chunk, call, staged[3], shape)
        # A gradient a scope, each done before the next is formed (`lax.optimization_barrier`), the rows'
        # scaled by their outer factor before the keys' is summed: no two held while products run.
        d_query = summed([(functools.partial(read, columns[0], scale), keys_staged)])
        d_keyed = summed([(functools.partial(read, columns[2], 1.0), keys_staged)])
        separated = algebra.separated(*limbs())
        outer = algebra.between(separated, algebra.own_anchors(at_token, shape, algebra.FIRST))
        d_rows = lax.optimization_barrier((outer * d_query, outer * d_keyed))
        d_key = summed([(functools.partial(read, columns[1], scale), queries_staged),
                        (functools.partial(read, columns[3], 1.0), gated_staged)])
        separated = algebra.separated(*limbs())
        last = algebra.own_anchors(at_token, shape, algebra.LAST)
        return d_rows, algebra.between(last, separated) * d_key

    return lax.cond(failed, exact, anchored)


def _by_sub_chunk(call: Call, tile: tuple, shape: tuple[int, int], terms: list) -> jax.Array:
    """The sum `[C, D]` of `terms`, each `(block, scaled)`: a compact block `[C, SUB]` (`algebra.own_columns`,
    read by `block()` once a term) against its rows' own sub-chunk's rows of `scaled` (`Parts`), a sub-chunk
    at a time: its rows of the block, the rest zero, staged in `tile` (`BLOCK_PARTS` parts, the last
    sub-chunk's products done reading it), against that sub-chunk's 16 rows (`Parts.rows`), into one
    accumulator in a scope of its own (`Family.summed`). The sub-chunks a loop (`lax.fori_loop`, not
    unrolled) where the family reads its operands' fragments in registers (mma.sync's, its sum a carry) or
    its accumulator is a warpgroup's (wgmma's, written in place): unrolled, ptxas hoisted their fragments'
    loads, a branch at 252 registers (row 5260)."""
    family, count = call.family, call.chunk // algebra.SUB

    def issue(acc):
        for block_of, scaled in terms:
            block = block_of()

            def sub_chunk(s, acc, block=block, scaled=scaled):
                family.settled(0)
                left = operands.staged(call, _sub_chunk(block, s), tile)
                rights = scaled.rows(s * algebra.SUB, algebra.SUB).refs
                return family.issued(acc, left.refs, rights, call.interpret)

            if call.interpret or not (family.operands_in_registers or family.name.startswith("wgmma")):
                for s in range(count):
                    acc = sub_chunk(s, acc)
            elif family.operands_in_registers:
                acc = lax.fori_loop(0, count, sub_chunk, acc, unroll=False)
            else:
                # The warpgroup's accumulator written in place by each product: no carry.
                def in_place(s, carry, run=sub_chunk, ref=acc):
                    run(s, ref)
                    return carry

                lax.fori_loop(0, count, in_place, None, unroll=False)
        return acc

    return family.summed(shape, issue)


def _sub_chunk(tile: jax.Array, s) -> jax.Array:
    """`tile` `[C, SUB]`'s rows of sub-chunk `s`, zero elsewhere."""
    rows = lax.broadcasted_iota(jnp.int32, tile.shape, 0) // algebra.SUB
    return jnp.where(rows == s, tile, 0.0)


def _columns(call: Call, at: tuple, ref, factor: float) -> jax.Array:
    """A pairs' gradient's compact block `[C, SUB]` (`algebra.own_columns`') read from `ref` at `at` (the
    chunk's row and the head's column), scaled by `factor`."""
    row, column = at
    (view,) = fp32_operand(call).views(ref, row, call.chunk, column, algebra.SUB)
    return call.family.tile_of(view) * factor


def back_refs(refs: tuple, staging: tuple, anchors: bool, *, outputs: int) -> tuple:
    """`key_pairs_own`'s program's `(columns, refs, anchored, staging)`: where its pairs are anchored products
    (`anchors`), the pairs' gradients' compact blocks (`COLUMNS`), its last inputs before its `outputs` (read
    by `back`), taken out of `refs`, and the anchored operands' staging, `staging`'s last."""
    if not anchors:
        return (), refs, (), staging
    given = len(refs) - outputs
    return (refs[given - COLUMNS:given], (*refs[:given - COLUMNS], *refs[given:]), staging[-1],
            staging[:-1])


def back_inputs(d_scores: jax.Array, d_system: jax.Array, *, chunk: int) -> tuple:
    """`back`'s compact blocks (`COLUMNS`, `algebra.own_columns`) of the pairs' gradients' squares, formed
    in XLA: a quarter of a square's bytes each."""
    return tuple(algebra.own_columns(d, chunk=chunk, transposed=transposed)
                 for d in (d_scores, d_system) for transposed in (False, True))
