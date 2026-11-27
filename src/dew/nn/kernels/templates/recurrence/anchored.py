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
from dew.nn.kernels.templates.recurrence.mainloop import Call, f32, staging_tiles


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
