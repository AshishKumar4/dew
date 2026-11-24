"""The attention template's Mosaic GPU dK/dV launch keys first (`attention_mosaic_backward`): each consumer's
key block resident, walked by the query tiles that read it, its dK and dV, and the hook's key and tile
gradients; the fused fast form's keys-first launch adds each tile's dQ too."""

import dataclasses
import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import launches, tuning
from dew.nn.kernels.templates import (
    across,
    attention_algebra as algebra,
    attention_gradient as gradient,
    mosaic,
    mosaic_sm100,
    outputs,
    placed,
    rings,
    stacked,
    stores,
    swizzling,
)
from dew.nn.kernels.templates.attention import DKV_GRADS, Dropout, Kind, Param, by_power
from dew.nn.kernels.templates.attention_mosaic_program import (
    BackwardTiles,
    TileWalk,
    _coords,
    _Spans,
    _streamed,
    _views_of,
    column_share,
    head_first,
    head_layout,
    head_slice,
    head_unlayout,
    held_params,
    holders,
    issued_into,
    modelled,
    param_types,
    placed_scratch,
    reducing,
    riding,
    scheduled_roles,
    score_streams,
    score_views,
    split_queries,
    stages_after_walks,
    stated,
    transposed_swizzles,
)
from dew.nn.kernels.templates.attention_mosaic_scores import (
    KEY_ROWS,
    _Backward,
    _dq_added,
    _relay_tiles,
    _relayed,
    _score_cotangents,
    _score_params,
    _score_weights,
    _scores,
    _splits_scores,
)


def _dkv_allocations(family: mosaic.Family, config: BackwardTiles, width: int, value_width: int, dtype, *,
                     stages: int, columns: int = 1, tiled: bool = False,
                     params: tuple = ()) -> tuple[list, dict, list]:
    """The dK/dV program's allocations as `Family.scoped` takes them, in `_dkv_launch`'s order: its consumers'
    resident keys and values; dK's output stage and dV's (the one stage where the widths agree), fp32, each
    its program's share of the `columns`; each consumer's stage for its key gradients' sums; the two loads'
    barriers; a fused launch's (`BackwardTiles.fused`) staged score cotangents and dQ tile each consumer's;
    where its logits' queries are staged (`BackwardTiles.staged_logits`) one slot's of them and their two
    turns; where a score hook's tile parameter is `tiled`, each consumer's stage for its tile gradients' adds
    (`Family.added`); each consumer's relay tile where its weights and score cotangents reach dV's and dK's
    products through shared memory (`Family.relay_bytes`, `_relayed`); and its query and output-cotangent
    streams, and its logits' queries' unless staged, in `stages` slots (`Family.streams`). On a layout whose
    consumers feed their rings (`split_queries`) the consumers share one key block, held once, and each takes
    its share of a slot's queries: every block a product of theirs reads staged at the swizzle their share's
    contraction takes (`swizzling.swizzle_for`). The output and key gradients' stages, which no walk reads,
    are scoped after the walks where the family scopes them apart (`stages_after_walks`), in their slots. The
    score hook's parameters that ride the stream of queries (`params`, each its `Kind` and dtype:
    `score_streams`: a tile's keys-first block of the program's key block) take one each, staged plain."""
    consumers, block_q, block_k = config.consumers, config.block_q, config.block_k
    split = split_queries(family)
    # A family placing its products on a role keys first (`placed.KEYS`) holds one key block its compute
    # warpgroups walk in turn, each a whole slot of queries a step.
    holders, rows = ((1, block_q // consumers) if split else (1, block_q) if family.places_on_role
                     else (consumers, block_q))
    swizzle = swizzling.swizzle_for(dtype, rows, width, value_width) if split else None
    # Each holder's output stages, where it stores its own sums: the consumers sharing a key block store
    # theirs summed once, in a scope of their own after the walk's (`_summed_stored`).
    # A family placing its products on a role reads its sums out of tensor memory a stage's columns at a time
    # (`placed.Tcgen05Placed.stored`, CUTLASS's sub-tiled epilogue): its stages that many columns (a fused
    # walk at width 128 staged 328,816 bytes with whole-width stages, past a B200's 232,448, et-keys10).
    out_width, value_out = width // columns, value_width // columns
    if family.places_on_role:
        out_width = value_out = stores.staged_columns(out_width, (jnp.float32,))
    outputs = [] if split else [
        family.staged_memory((holders, max(block_q, block_k), out_width), jnp.float32),
        *([] if value_width == width
          else [family.staged_memory((holders, block_k, value_out), jnp.float32)]),
        *family.vector_stage((holders, block_k), jnp.float32)]
    after = outputs if stages_after_walks(family) else []
    scoped = [family.staged_memory((holders, block_k, width), dtype, swizzle),
              family.staged_memory((holders, block_k, value_width), dtype, swizzle),
              *([] if after else outputs), family.load_barriers(holders), family.load_barriers(holders)]
    blocks: dict[str, tuple] = {"queries": ((block_q, width), dtype),
                                "douts": ((block_q, value_width), dtype)}
    if config.fused:
        # A fused launch's (`_dkv_step`'s `add_dq`): each consumer's score cotangents staged for dQ's
        # product, their transpose its left operand, and its dQ tile's stage for the add; a placed walk's
        # a ring of `REDUCED` slots of half a stage's columns each, a barrier a slot for its fill (the
        # compute warpgroup's arrival) and one for its free (the reducer's), `_PlacedDq`.
        scoped.append(family.staged_memory((consumers, block_k, rows), dtype))
        if family.places_on_role:
            scoped += [family.staged_memory((consumers, REDUCED, rows, out_width // REDUCED), jnp.float32),
                       *(tuple(plgpu.Barrier(num_arrivals=1, num_barriers=REDUCED) for _ in range(consumers))
                         for _ in range(2))]
        else:
            scoped.append(family.staged_memory((consumers, rows, width), jnp.float32))
    if config.staged_logits:
        # One slot's queries taken by the power, which the first consumer stages for every consumer's share,
        # and the two turns that order its writes against their products' reads (`_staged_logits`): the
        # logits' product reads that stage transposed, at the swizzle a transposed read takes.
        staged = transposed_swizzles(config, "logits").get("logits", swizzle) if swizzle is None else swizzle
        scoped += [family.staged_memory((block_q, width), dtype, staged), family.schedule(consumers),
                   family.schedule(consumers)]
    else:
        blocks["logits"] = ((block_q, width), dtype)
    if tiled:
        scoped.append(family.staged_memory((consumers, block_k, rows), jnp.float32))
    scoped += _relay_tiles(family, consumers, block_k, rows, dtype)
    # Keys first, the logits' and dP's products read their queries (the stream of them by the power where it
    # has one, its stage where staged) and the output cotangents transposed; a launch's own swizzle every
    # stream's.
    transposed = transposed_swizzles(config, *(("logits",) if "logits" in blocks else ()), "douts")
    swizzles = transposed if swizzle is None else dict.fromkeys(blocks, swizzle)
    # A tile parameter's blocks of one key block's, the one holder a launch streaming them takes
    # (`MosaicBackward.legal`).
    hooked = score_streams(family, params, "queries", block_k, block_q)
    swizzles = {**swizzles, **dict.fromkeys(hooked, swizzling.UNSWIZZLED),
                **(placed.walked_swizzles(KEYS_FIRST, dtype, block_q) if family.places_on_role else {})}
    # A walk placed on a role holds a stream its ahead products alone read in one slot
    # (`placed.ahead_alone`), loaded last of a step's.
    once = ({name: blocks.pop(name) for name in placed.ahead_alone(KEYS_FIRST)}
            if family.places_on_role else {})
    streams = family.streams({**blocks, **hooked}, stages, consumers, swizzles=swizzles)
    if once:
        streams = {**streams, **family.streams(once, 1, consumers, swizzles=swizzles)}
    return scoped, streams, after


def _dkv_step(walk: _Backward, resident: tuple, rows_of, coords_of, params_of, store_tiles, shares, part,
              logits_stage, add_dq, index, slots, carry, accumulators, *, masked: bool = True):
    """A dK/dV consumer's query tile `index`, its keys the tile's rows (`[bk, bq]`, FlashAttention 3's order,
    hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp's SdP_swapAB): its scores `K Qᵀ` (the call's power taken after
    the product) and score cotangents from the query and output cotangent slots (`slots`) and its resident
    keys and values (`resident`), under the tile's queries' statistics (`rows_of(first query)`, in a column's
    layout), its coordinates (`coords_of(first query)`) and its hook parameters (`params_of(first query,
    like, slots=slots)`, those riding the queries' stream the slots'). The weights and the score cotangents,
    their keys as their rows, are dV's and dK's left operands from the registers (`algebra.value_operands`,
    `cotangent_product`'s `aᵀ @ b`), issued into `accumulators` in place. The key gradients' sums (`carry`)
    take the pulled cotangents' row sums, and the tile gradients go to `store_tiles(first query, pieces)`,
    `[bk, bq]` each. The consumer takes `part` of each slot's queries (`(first, count)`, `split_queries`; the
    whole slot where its keys are its own)."""
    k_smem, v_smem = resident
    family, tiles, kinds = walk.program.family, walk.program.tiles, walk.kinds
    first_row, count = part
    shape = (tiles.block_k, count)
    dk_acc, dv_acc = accumulators
    q_slot, do_slot = slots["queries"], slots["douts"]
    if logits_stage is not None:
        logits_slot = _staged_logits(walk, logits_stage, q_slot, part)
    else:
        logits_slot = slots["logits"]
        logits_slot = logits_slot if count == tiles.block_q else logits_slot.at[pl.ds(first_row, count)]
    if count != tiles.block_q:
        q_slot, do_slot = q_slot.at[pl.ds(first_row, count)], do_slot.at[pl.ds(first_row, count)]
    q_first = index * tiles.block_q + first_row
    # The logits from the queries taken by the call's power, the forward's `(Q 2^p) Kᵀ` bit for bit (the
    # power taken after the product passes fp32's largest where the forward's does not, Q 2^120 over keys of
    # 8 at width 64 and scale 1/8; on the keys a key's entry can leave bf16's range, 2^-126 at the scale
    # 2^-10): a stream of their own, or where one passes the block staged a step from the call's own
    # (`BackwardTiles.staged_logits`; a third of the launch at Qwen3-8B, ET-SWAPABL-h100). dK from the call's
    # own, which the power would flush where a query is 2^-126 at the scale 2^-10.
    read = walk.places_read(masked=masked)
    fused = add_dq is not None
    coords = coords_of(q_first, shape=shape, read=read)
    rows = rows_of(q_first, count)
    dtypes = (do_slot.dtype, q_slot.dtype)
    if _splits_scores(walk):
        def between(s):
            if logits_stage is not None:
                walk.program.family.turn(logits_stage[2], walk.consumers, wait=False)
            _, weights, slopes, seen = _score_weights(walk, coords, (), s, rows[0], rows[1], masked=masked)
            return weights, slopes, seen

        # S and dP issued together, S's weights made as dP runs on (`Issuer.summed_together`'s `then`).
        (weights, slopes, seen), dp = (read() for read in family.summed_together(
            (shape, shape), (lambda acc: family.products(acc, k_smem, logits_slot.transpose((1, 0))),
                             lambda acc: family.products(acc, v_smem, do_slot.transpose((1, 0)))),
            then=(between, None)))
        cotangents = _score_cotangents(coords, weights, dp, rows[2], rows[3], seen)
        scored = (weights, weights, cotangents, slopes, seen)
    else:
        s, dp = (read() for read in family.summed_together(
            (shape, shape), (lambda acc: family.products(acc, k_smem, logits_slot.transpose((1, 0))),
                             lambda acc: family.products(acc, v_smem, do_slot.transpose((1, 0))))))
        if logits_stage is not None:
            # Staged queries read (products waited): next step's stage may be written.
            walk.program.family.turn(logits_stage[2], walk.consumers, wait=False)
        # run time joins both choices' tiles and its column statistics, 16 registers each, past a consumer's
        # (spilled 1184 bytes at 64 x 64, one consumer).
        scored = _scores(walk, coords, params_of(q_first, s, slots=slots), s, dp, *rows, masked=masked)
    narrowed, logits, pieces = _weighed(walk, coords, scored, dtypes, one_part=fused)
    (q0, q_share), (v0, v_share) = shares
    q_columns, do_columns = q_slot.at[:, pl.ds(q0, q_share)], do_slot.at[:, pl.ds(v0, v_share)]
    dv_acc = issued_into(family, dv_acc, narrowed if isinstance(narrowed, tuple) else (narrowed,), do_columns,
                         walk.interpret)
    if not fused:
        # Each part formed as its product is issued (`algebra.formed_parts`).
        parts = algebra.formed_parts(logits * walk.grad_scale, q_slot.dtype, accurate=walk.accurate,
                                     rows=True)
        dk_acc = issued_into(family, dk_acc, parts, q_columns, walk.interpret)
    else:
        # The fused launch's (the fast form's one part, `_weighed`'s): dK's left operand from the registers,
        # and staged, its transpose dQ's (`algebra.cotangent_product`'s `a @ b`), the tile's dQ added into
        # dQ, the query tile's other key blocks adding theirs (`_fused_adds`); a consumer's share of a
        # 128-query slot is 64 rows, warpgroup MMA's least (`MosaicBackward.legal`).
        dk_acc = issued_into(family, dk_acc, (logits,), q_columns, walk.interpret)
        add_dq(logits, q_first)
    if _grading(shares):
        store_tiles(q_first, [piece for kind, piece in zip(kinds, pieces, strict=True) if kind == Kind.TILE])
        key_pieces = [piece for kind, piece in zip(kinds, pieces, strict=True) if kind == Kind.KEY]
        carry = tuple(total + piece for total, piece in zip(carry, key_pieces, strict=True))
    return carry, (dk_acc, dv_acc)

PART = 16
"""The queries a compute warpgroup of a placed keys-first walk takes of a step at once (`_placed_step`): a
thread's key row's scores and their cotangents, 32 registers, beside the step's statistics of them."""

def dq_halves(width: int) -> tuple[int, ...]:
    """A fused keys-first walk's dQ's columns as its turns hold them (`placed.DQ`): a 64-row accumulator of
    at most 128 columns over each turn of a step, in their order (width 256 over both)."""
    return tuple(min(128, width - first) for first in range(0, width, 128))


FUSED_DQ = (mosaic.Product("keys", placed.DQ, "staged"),)
"""A fused keys-first walk's dQ product (`placed.WalkedProducts.fused`): `dS K` each step, its left the
step's score cotangents its compute warpgroup stages, its right the resident keys (`_PlacedDq`)."""

KEYS_FIRST = (mosaic.Product("logits", placed.SCORES, "keys", transposed=True, ahead=True),
              mosaic.Product("douts", placed.COTANGENTS, "values", transposed=True, ahead=True),
              mosaic.Product("douts", placed.DV, placed.SCORES, accumulate=True),
              mosaic.Product("queries", placed.DK, placed.COTANGENTS, accumulate=True))
"""A keys-first walk's products as a family placing them on a role issues them (`placed.walked_products`,
FlashAttention 4's backward): its scores `Sᵀ = K Qᵀ` from the queries taken by the call's power and their
cotangents `dPᵀ = V dOᵀ`, a step ahead, and dV's and dK's onto their sums, their lefts the weights and score
cotangents the step's compute warpgroup writes over the scores' and cotangents' turns (`_placed_step`)."""


def _placed_step(walk: _Backward, rows_of, coords_of, dq_added, index, slots, carry, accumulators, *,
                 masked: bool = True):
    """A compute warpgroup's query tile `index` of a walk whose family places its products on a role keys
    first (`placed.KEYS`, `KEYS_FIRST`): the tile's scores and their cotangents its issuer's (`awaited`, a key
    a thread's row in tensor memory's own layout), `PART` queries at a time under their statistics
    (`rows_of(first query, count)`) and coordinates (`coords_of`), the algebra's weights and pulled score
    cotangents (`_weighed`) written narrowed over the part's own columns of the turn (FlashAttention 4's `P`
    over `S`, a part's 16-bit values over its first half, read already), then given to the issuer for dV's
    and dK's products (`given`). The plain call's alone: no hook's parameters, dropout or tile gradients
    (`MosaicBackward._keys_first_legal`). A fused walk's (`dq_added`, `_PlacedDq`) stages its score
    cotangents for its step's dQ product too, and adds that dQ into the call's once the issuer's is done."""
    family, tiles = placed.walked(walk.program.family), walk.program.tiles
    q_first = index * tiles.block_q
    scores, cotangents = family.awaited(slots)
    weights = family.turn_of(placed.SCORES, slots.turn, narrowed=True)
    pulled = family.turn_of(placed.COTANGENTS, slots.turn, narrowed=True)
    read = walk.places_read(masked=masked)
    for first in range(0, tiles.block_q, PART):
        s = family.read(scores, first=first, columns=PART, layout=mosaic_sm100.TMEM)
        dp = family.read(cotangents, first=first, columns=PART, layout=mosaic_sm100.TMEM)
        coords = coords_of(q_first + first, shape=(tiles.block_k, PART), read=read)
        scored = _scores(walk, coords, (), s, dp, *rows_of(q_first + first, PART), masked=masked)
        (narrowed,), logits, _ = _weighed(walk, coords, scored, (weights.dtype, pulled.dtype), one_part=False)
        family.write(weights, mosaic.unscaled_part(narrowed), first=first)
        (cotangent,) = algebra.formed_parts(logits * walk.grad_scale, pulled.dtype, accurate=False, rows=True)
        family.write(pulled, mosaic.unscaled_part(cotangent), first=first)
    if dq_added is not None:
        # The step's score cotangents staged for its dQ product from tensor memory in the layout a swizzled
        # tile takes (`mosaic_sm100.STORED`): tensor memory's own, a thread a row, stores to none (Mosaic
        # GPU's layout inference refused the parts' stores, et-fused-lower).
        plgpu.commit_tmem()
        dq_added.staged(slots.turn)[...] = family.read(pulled, layout=mosaic_sm100.STORED)
        plgpu.commit_smem()
    family.given(slots)
    if dq_added is not None:
        dq_added.added(family.dq_read(slots), q_first, slots)
    return carry, accumulators


REDUCED = 2
"""The slots of each compute warpgroup's ring of a placed fused walk's dQ (`_PlacedDq`): it writes one while
the reducer's add reads the other (`stores.OUTPUT_BUFFERS`'s two), each half the one stage's columns it
replaces, so the launch's shared memory is unchanged."""


@dataclasses.dataclass(frozen=True)
class _PlacedDq:
    """A fused keys-first walk's dQ (`_placed_step`'s `dq_added`): each compute warpgroup's stage of its
    step's score cotangents (`staged(turn)`, `[bk, bq]`) and its ring of `REDUCED` dQ slots, which it fills
    a slot's columns of its step's dQ at a time (`added`) and the launch's reducer adds into the call's at
    the step's query tile (`reduced`, `handoffs.REDUCER`), each slot's fill and its free a barrier's phase
    (the ring's `filled`, the warpgroup's arrival, and `emptied`, the reducer's, once its add has read the
    slot). A compute warpgroup waits on no add of its own: Dew's dK/dV on a B200 spent its compute
    warpgroups' time waiting on their own adds' reads before each next chunk's write (row 8283)."""

    walk: _Backward
    dq_ref: Any
    stages: tuple
    place: tuple

    def staged(self, turn: int):
        return self.stages[0].at[turn]

    def pieces(self) -> tuple[tuple[int, int], ...]:
        """A step's dQ a slot's columns at a time, in turn, each its first column and its count: its halves'
        (`dq_halves`), sliced by `stores.column_spans` (one chunk plan shared with `added`, row 8813)."""
        columns, found, first = self.stages[1].shape[-1], [], 0
        for half in dq_halves(self.dq_ref.shape[-1]):
            found += [(first + at, count) for at, count in stores.column_spans(half, columns)]
            first += half
        return tuple(found)

    def _turn(self, ordinal, piece: int):
        """Piece `piece` of a compute warpgroup's `ordinal`-th step's turn of its ring."""
        return rings.Turn.of(ordinal * len(self.pieces()) + piece, REDUCED)

    def _view(self, q_first, turn: int = 0):
        row, head = self.place
        return self.dq_ref.at[turn, row, pl.ds(q_first, self.walk.program.tiles.block_q), head]

    def added(self, halves: tuple, q_first, slots) -> None:
        """A compute warpgroup's step's dQ `halves` (`slots`' turn its role, its ordinal among its steps the
        walk's over the turns) written into its ring a slot at a time, each slot once the reducer's add has
        read its last fill (`emptied`), its writes fenced for the add's read and the fill published
        (`filled`); interpreted, added into its own copy of the call's (`Family.added`, no reducer)."""
        family, stage = self.walk.program.family, self.stages[1]
        filled, emptied = self.stages[2][slots.turn], self.stages[3][slots.turn]
        if self.walk.interpret:
            first = 0
            for half in halves:
                columns = half.shape[1]
                family.added(half, self._view(q_first, slots.turn).at[:, pl.ds(first, columns)],
                             interpret=True, smem=stage.at[slots.turn, 0])
                first += columns
            return
        ordinal = slots.n // placed.TURNS
        written = [piece for half in halves for _, piece in mosaic.column_chunks(half, stage.shape[-1])]
        for index, piece in enumerate(written):
            turn = self._turn(ordinal, index)
            before = rings.Turn.of(turn.counter - REDUCED, REDUCED)
            pl.when(turn.counter >= REDUCED)(lambda before=before, slot=turn.slot: plgpu.barrier_wait(
                emptied.at[slot], phase=before.rotation))
            stage.at[slots.turn, turn.slot][...] = piece
            plgpu.commit_smem()
            plgpu.barrier_arrive(filled.at[turn.slot])

    def reduced(self, lo, hi, role: int) -> None:
        """A compute warpgroup's (`role`'s) reducer: the adds of its steps among the walk's `lo` to `hi` (step
        `n` the warpgroup's of its turn, `placed.TURNS`), each piece once filled (`filled`), each slot freed
        once the add after it is issued and its own read done (`emptied`), two adds reading at once; its last
        freed and its adds done before the program ends. A reducer a role: one serving both took each
        role's whole step in turn, the other's filling waiting on it. Interpreted, nothing (`added`)."""
        if self.walk.interpret:
            return
        stage, filled, emptied = self.stages[1], self.stages[2][role], self.stages[3][role]
        block_q, pieces = self.walk.program.tiles.block_q, self.pieces()

        def freed(last) -> None:
            # The add before the one just issued read: its slot free (the walk's first has none before it).
            pl.when(last >= 0)(lambda: plgpu.barrier_arrive(emptied.at[last]))

        def step(ordinal, last):
            view = self._view((lo + role + placed.TURNS * ordinal) * block_q)
            for index, (first, columns) in enumerate(pieces):
                turn = self._turn(ordinal, index)
                plgpu.barrier_wait(filled.at[turn.slot], phase=turn.rotation)
                plgpu.copy_smem_to_gmem(stage.at[role, turn.slot], view.at[:, pl.ds(first, columns)],
                                        reduction_op="add")
                plgpu.wait_smem_to_gmem(1, wait_read_only=True)
                freed(last)
                last = jnp.asarray(turn.slot, jnp.int32)
            return last

        steps = lax.div(jnp.maximum(hi - lo - role, 0) + placed.TURNS - 1, placed.TURNS)
        last = lax.fori_loop(0, steps, step, jnp.int32(-1))
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        freed(last)
        plgpu.wait_smem_to_gmem(0)

    def drained(self, lo, hi, role: int) -> None:
        """A compute warpgroup's (`role`'s) last `REDUCED` fills of its ring read by the reducer's adds
        (`emptied`), its walk's steps `lo` to `hi` the walk's in turn: no slot's free left unobserved, and
        no scope after the walk written over a slot an add still reads."""
        if self.walk.interpret:
            return
        steps = lax.div(jnp.maximum(hi - lo - role, 0) + placed.TURNS - 1, placed.TURNS)
        rings.drained((self.stages[3][role],), steps * len(self.pieces()), REDUCED)


def _weighed(walk: _Backward, coords: algebra.Coords, scored: tuple, dtypes: tuple, *,
             one_part: bool) -> tuple:
    """A keys-first tile's: its weights narrowed for dV in the output cotangents' dtype
    (`algebra.value_weights`, the dropped ones, `algebra.drawn`; the dQ program's sums take them
    undropped), its pulled score cotangents and its hook's pieces (`gradient.pulled`), from its `_scores`
    5-tuple `scored`; the cotangents the fast form's one part in the queries' dtype where the launch takes
    it from the registers (`one_part`, the fused launch's), narrowed as they are formed."""
    values, queries = dtypes
    exponentials, weights, cotangents, slopes, seen = scored
    logits, pieces = gradient.pulled(cotangents, slopes, seen, walk.kinds, KEY_ROWS)
    if one_part:
        parts = algebra.cotangent_parts(logits * walk.grad_scale, queries, accurate=False)
        logits = mosaic.unscaled_part(*parts)
    multiplier = algebra.drawn(coords, coords.q_pos, coords.k_pos)
    if multiplier is not None:
        exponentials, weights = exponentials * multiplier, weights * multiplier
    if one_part:
        # The fused launch stages them for dV's product (bf16 alone, `MosaicBackward.legal`).
        return algebra.value_weights(exponentials, weights, values, multiplier is not None), logits, pieces
    return algebra.value_parts(weights, values), logits, pieces


def _staged_logits(walk: _Backward, stage: tuple, q_slot, part: tuple):
    """A step's logits' queries, the slot `q_slot`'s taken by the call's power, staged in `stage` (its ref,
    its written and read turns, the consumer's index): the first consumer stages the whole slot (a share of
    32 of its 64 rows takes no warpgroup-MMA tile to load and store), once every consumer has read the last
    step's (`read`, which each arrives at past its product and once before its walk, and waits for once
    past it), and every consumer waits for it (`written`); the consumer's share of it, `part`'s rows."""
    ref, written, read, index = stage
    family, consumers = walk.program.family, walk.consumers
    family.turn(read, consumers, arrive=False)

    @pl.when(index == 0)
    def _():
        family.staged(by_power(family.tile_of(q_slot), walk.spec.prescale), ref)

    family.turn(written, consumers)
    first, count = part
    return ref if count == ref.shape[0] else ref.at[pl.ds(first, count)]

def _fused_adds(walk: _Backward, dq_ref, stages: list, place: tuple, index, k_smem, holder):
    """A fused launch's consumer `index`'s dQ adds (`_dkv_step`'s `add_dq(score cotangents, first query)`),
    None where the launch adds none: its score cotangents staged (`stages[0]`), their transpose times its
    keys (`k_smem`, `[holders, bk, D]`, its `holder`'s), the tile's dQ added into dQ at `place`
    (`_dq_added`)."""
    if not walk.fused:
        return None
    family = walk.program.family
    staged, out_stage = stages

    def own(part, q_first):
        family.staged(part, staged.at[index])
        held = k_smem.at[holder]
        _dq_added(walk, dq_ref, out_stage, place, index, q_first,
                  family.product(staged.at[index].transpose((1, 0)), held, (part.shape[1], held.shape[-1])))

    return own

def _logits_stage(family: mosaic.Family, consumers: int, allocated: list, index) -> tuple | None:
    """Consumer `index`'s staged logits' queries (`_staged_logits`' `stage`) from their `allocated` ref and
    turns (`Family.schedule`), its read turn's first arrival made (no step's stage is written before it);
    None where its launch streams them."""
    if not allocated:
        return None
    family.turn(allocated[2], consumers, wait=False)
    return (*allocated, index)

def _grading(shares: tuple) -> bool:
    """Whether a dK/dV program at its head widths' `shares` (`BackwardTiles.columns`) writes the score
    parameters' gradients: the first share's alone, each share's logits and their cotangents the tile's
    whole."""
    return shares[0][0] == 0


def _tiles_added(family: mosaic.Family, refs, stage, place: tuple, k_first, index, interpret: bool, first,
                 pieces) -> None:
    """Each tile gradient of `pieces` (`[bk, its queries]`, queries from `first`) added into its place in the
    keys-first `[B, H, Kp, Qp]` refs (`refs`), the program's row's and head's (`place`, `_grad_buffers`),
    through consumer `index`'s `stage`."""
    row, head = place
    for ref, piece in zip(refs, pieces, strict=True):
        assert stage is not None
        view = ref.at[row, head, pl.ds(k_first, piece.shape[0]), pl.ds(first, piece.shape[1])]
        family.added(piece, view, interpret, smem=stage.at[index])


def _dkv_stored(family: mosaic.Family, sums: tuple, views: tuple, stages: tuple, *, index, interpret: bool
                ) -> None:
    """A dK/dV consumer's dK and dV of its own keys to their `views`, and its key gradients' sums added into
    theirs (`Family.added`, the program's row's and head's zeros), or written where its family adds no
    vector (`Family.adds_vectors`), through its `stages` (dK's output stage, dV's, the key gradients')."""
    out_smem, dv_smem, rows_smem = stages
    dk, dv, *key_sums = sums
    dk_view, dv_view, *key_views = views
    dk_stage = out_smem.at[index, pl.ds(0, dk.shape[0])]
    if dv_smem is out_smem:
        family.stored((dk, dv), (dk_view, dv_view), smem=dk_stage)
    else:
        family.stored((dk,), (dk_view,), smem=dk_stage)
        family.stored((dv,), (dv_view,), smem=dv_smem.at[index])
    rows_stage = None if rows_smem is None else rows_smem.at[index]
    for key_sum, key_view in zip(key_sums, key_views, strict=True):
        # Each key gradient, the walk's row sums, added into its row's and head's `[B, H, Kp]`.
        if family.adds_vectors:
            family.added(key_sum, key_view, interpret, smem=rows_stage)
        else:
            family.stored((family.as_row(key_sum),), (key_view,), smem=rows_stage)


def _grad_buffers(graded: list[Kind], batch: int, heads: int, queries: int, keys_padded: int) -> list[tuple]:
    """Each graded parameter's gradient buffer's shape (`_dkv_launched`): the call's batch rows and heads
    whatever rows and heads the parameter shares, so a row's part lands at its own scale (the values it was
    taken beneath, `template.value_shift`) and `template.gradients` sums the shared rows after, each at its
    own (`template.hook_gradient`), then its keys, and for a tile its queries."""
    return [(batch, heads, keys_padded) if kind == Kind.KEY else (batch, heads, keys_padded, queries)
            for kind in graded]

def _graded(kinds: tuple, shares: tuple, shape: tuple, keys_padded: int
            ) -> tuple[list[Kind], list[tuple], tuple[tuple[int, ...], ...]]:
    """The kinds of the hook gradients a dK/dV program of `shares` writes (`DKV_GRADS`; none for a later
    share's), their buffers' shapes (`_grad_buffers`) and the groups of one shape (`stacked.groups`),
    for a call whose queries are `shape` `[B, Q, H, D]`."""
    graded = [kind for kind in kinds if kind in DKV_GRADS] if _grading(shares) else []
    buffers = _grad_buffers(graded, shape[0], shape[2], shape[1], keys_padded)
    return graded, buffers, stacked.groups(buffers)


def _dkv_refs(refs, treedef, leaves: int, graded: list[Kind], groups: tuple, *, widths: bool, split: bool,
              fused: bool, after: bool, vectors: bool) -> tuple:
    """The dK/dV program's refs by role: its nine inputs, its spans' and hook parameters' (`treedef` of
    `leaves`), a fused launch's dQ and the hook's key and tile gradients, each added into in place, a ref a
    group of equal shapes (`_dkv_launched`, `stacked.groups`), its dK and dV outputs, and its scoped
    allocations (`_dkv_scoped`)."""
    inputs, refs = refs[:9], refs[9:]
    unflattened, refs = treedef.unflatten(refs[:leaves]), refs[leaves:]
    dq_ref, refs = (refs[0], refs[1:]) if fused else (None, refs)
    grad_refs, refs = refs[:len(groups)], refs[len(groups):]
    views = stacked.views(grad_refs, groups)
    dk_ref, dv_ref, *refs = refs
    keyed = tuple([ref for ref, kind in zip(views, graded, strict=True) if kind == wanted]
                  for wanted in DKV_GRADS)
    held = _dkv_scoped(refs, widths=widths, staged=not split and not after, vectors=vectors)
    return inputs, unflattened, (dq_ref, dk_ref, dv_ref), held, keyed

def _shared_views(family: mosaic.Family, outputs, grad_refs, graded: list[Kind], block_k: int) -> tuple:
    """dK's and dV's views (`outputs`, the dK/dV launch's) and each key gradient's (`grad_refs`, the program's
    row's and head's, `_grad_buffers`) at the program's key block, the one its consumers share
    (`split_queries`)."""
    dk_ref, dv_ref = outputs
    row, head = family.program_index("batch"), family.program_index("heads")
    keys = pl.ds(family.program_index("k_tiles") * block_k, block_k)
    key_views = [ref.at[row, head, keys]
                 for ref, kind in zip(grad_refs, graded, strict=True) if kind == Kind.KEY]
    return dk_ref.at[row, keys, head], dv_ref.at[row, keys, head], *key_views

def _summed_stored(family: mosaic.Family, sums: tuple, views: tuple, consumers: int, interpret: bool) -> None:
    """The sums of the `consumers` sharing a key block (each over its share of the queries) added in a fixed
    order and stored once to their `views` (`across.summed_across`), dK's and dV's, the key gradients' added
    into theirs (`Family.added`, the program's row's and head's zeros), each in a
    scope after the walk's, whose shared memory it reuses (`across.summing`): the walk's last adds' reads of
    their stage (`Family.added`, its tile gradients') done first (row 2633, ET-DETDIAG-h100: the last query
    tile's gradient read from a stage this scope had written)."""
    family.store_wait(0)
    for place, (value, view) in enumerate(zip(sums, views, strict=True)):
        def written(summed, stage, view=view, place=place):
            if place < 2:
                family.stored((summed,), (view,), smem=stage)
            else:
                family.added(summed, view, interpret, smem=stage)

        def total(stage, value=value, written=written, **scratch):
            across.summed_across(value, scratch, lax.axis_index("wg"), consumers,
                                 lambda summed: written(summed, stage))

        pl.run_scoped(total, family.staged_memory(value.shape, jnp.float32),
                      **across.summing(value.shape, consumers), collective_axes="wg")

def _dkv_scoped(refs, *, widths: bool, staged: bool, vectors: bool) -> tuple:
    """The dK/dV program's allocations (`_dkv_allocations`) by role: its resident keys and values, dK's output
    stage and dV's (dK's where the head `widths` agree) and its key gradients' stage where the program's
    scope holds them (`staged`: none where its consumers share its keys, `split_queries`, or a scope after
    the walks holds them, `_output_stages`), its two loads' barriers and its logits' stage where it has
    one."""
    keys, values, refs = refs[0], refs[1], refs[2:]
    if not staged:
        return keys, values, None, None, None, *refs
    count = (1 if widths else 2) + vectors
    dk_stage, dv_stage, rows_stage = _output_stages(refs[:count], widths=widths, vectors=vectors)
    return keys, values, dk_stage, dv_stage, rows_stage, *refs[count:]


def _output_stages(refs, *, widths: bool, vectors: bool) -> tuple:
    """dK's output stage, dV's (dK's where the head `widths` agree) and the key gradients' from the first of
    `refs` (`_dkv_allocations`' outputs)."""
    dk_stage = refs[0]
    dv_stage = refs[0] if widths else refs[1]
    rows_stage = refs[1 if widths else 2] if vectors else None
    return dk_stage, dv_stage, rows_stage
def _keys_last(arrays, kinds) -> tuple[jax.Array, ...]:
    """Each tile kind's of `arrays` (a score hook parameter's or gradient's, `[B, H, ., .]`) with its last two
    axes swapped, between the call's queries-first entries and the dK/dV program's keys-first ones."""
    return tuple(jnp.swapaxes(array, -1, -2) if kind == Kind.TILE else array
                 for array, kind in zip(arrays, kinds, strict=True))

def _dkv_kernel(walk: _Backward, entry, graded: list[Kind], query, key, value, dout, shares: tuple,
                name: str):
    """The dK/dV program `entry`'s launch (`Family.kernel`) over `query` `[B, Qp, H, D]` and `key` and
    `value` `[B, Kp, Hk, D]`: dK and dV `[B, Kp, H, D]` fp32 out, each its share's columns (`shares`), the
    hook's key and tile gradients added into refs of their own (`_dkv_launched`); a program a query head's
    key block a consumer, or one its consumers share (`holders`)."""
    batch, queries, heads, width = query.shape
    keys_padded, value_width = key.shape[1], value.shape[-1]
    family, consumers, tiles = walk.program.family, walk.consumers, walk.program.tiles
    del graded
    # Each a query head's (`outputs.Output`): the query's batch rows and heads, the key's padded keys, a share
    # of the columns.
    rows, keys, heads_of = outputs.axis(query, 0), outputs.axis(key, 1), outputs.axis(query, 2)
    dkv_shape = (rows, heads_of, keys) if head_first(family, jnp.float32) else (rows, keys, heads_of)
    dk, dv = (outputs.output(jnp.float32, *dkv_shape, column_share(array, share))
              for array, (_, share) in zip((key, value), shares, strict=True))
    return family.kernel(
        entry, out_type=(dk, dv),
        grid=(heads, batch, keys_padded // (holders(family, consumers) * tiles.block_k)),
        grid_names=("heads", "batch", "k_tiles"), consumers=consumers, interpret=walk.interpret,
        approx_math=walk.approx_math,
        name=name,
        reducing_warpgroups=reducing(family, consumers, walk.reduces),
        registers=walk.registers(dq=False),
        # Its walk's statement, its work its emission's (`stated`); the model's products it owns
        # (`modelled`): its shares of dK's and dV's, dP's in the first share's and dQ's where it adds it.
        cost=stated(walk.spec, batch, heads, query, key, value, dout,
                    tile_walk=TileWalk((queries, keys_padded), (tiles.block_q, tiles.block_k),
                                       holders(family, consumers), keyed=True,
                                       common=family.feeds_itself or tiles.staged_logits)),
        model_flops=modelled(walk.spec, batch, heads, walk.spec.queries or queries,
                             (width if walk.fused else 0) + shares[0][1] + shares[1][1]
                             + (value_width if shares[0][0] == 0 else 0)))

def _dkv_launch(query, key, value, dout, peak, total, anchor, delta, key_index, spans: _Spans | None,
                score_params: tuple, drop: Dropout | None, scaled: jax.Array, *, walk: _Backward
                ) -> tuple[jax.Array, ...]:
    """`_dkv_columns`' launch of each of the config's `columns`, its share of dK's and dV's columns (the
    score hook's key and tile gradients the first's), dK and dV joined along their widths; `scaled` the
    queries taken by the call's power, its logits' (`_dkv_step`)."""
    columns = walk.program.tiles.columns
    width, value_width = query.shape[-1] // columns, value.shape[-1] // columns
    name = "attention_dkv_fused" if walk.fused else "attention_dkv"
    launched: list[tuple[jax.Array, ...]] = [
        _dkv_columns(query, key, value, dout, peak, total, anchor, delta, key_index, spans, score_params,
                     drop, scaled, walk=walk,
                     shares=((column * width, width), (column * value_width, value_width)),
                     name=name + ("" if columns == 1 else f"_columns{column}"))
        for column in range(columns)]
    if columns == 1:
        return launched[0]
    dk, dv = (jnp.concatenate([out[place] for out in launched], axis=-1) for place in (0, 1))
    return dk, dv, *launched[0][2:]

def _keys_stored(family: mosaic.Family, sums: tuple, views: tuple, stages: tuple, after: Callable | None, *,
                 index, widths: bool, interpret: bool) -> None:
    """A consumer's dK, dV and key gradients `sums` stored to their `views` (`_dkv_stored`) through the
    program's `stages`, or through those a scope `after` the walks holds (`stages_after_walks`, dK's and dV's
    one where the head `widths` agree, `_output_stages`)."""
    stored = functools.partial(_dkv_stored, family, sums, views, index=index, interpret=interpret)
    if after is None:
        stored(stages)
    else:
        after(lambda *scoped: stored(_output_stages(scoped, widths=widths, vectors=family.stages_vectors)))


@tuning.lowered_once(("walk", "shares", "name"))
def _dkv_columns(query, key, value, dout, peak, total, anchor, delta, key_index, spans: _Spans | None,
                 score_params: tuple, drop: Dropout | None, scaled: jax.Array, *, walk: _Backward,
                 shares: tuple, name: str) -> tuple[jax.Array, ...]:
    """The dK/dV program under the dQ program's softmax: each consumer's `block_k` keys of a query head,
    resident, walked by the query tiles that see them, streamed with their output cotangents through a ring
    each; a tile's weights and score cotangents (the algebra's, rows the keys, `_dkv_step`) the transposed
    products' left operands from the registers (`algebra.value_operands`, `cotangent_product`). Returns the
    query head's dK and dV `[B, Kp, H, D]` fp32, then each key and tile gradient of the score hook in kind
    order (`DKV_GRADS`: keys `[B, H, Kp]`, tiles `[B, H, Qp, Kp]`, a tile parameter read and its gradient
    written keys first). A key or tile gradient is written by the one program that holds its keys; the
    tiles its walk does not read are zeros. Its dK and dV are its `shares` of their columns (`(first, count)`
    each); a later share's program writes no hook gradient."""
    _, queries, heads, width = query.shape
    kv_heads, value_width = key.shape[2], value.shape[-1]
    program, consumers, kinds, family = walk.program, walk.consumers, walk.kinds, walk.program.family
    block_q, block_k = program.tiles.block_q, program.tiles.block_k
    split, placing = split_queries(family), family.places_on_role
    pair, count = holders(family, consumers) * block_k, block_q // consumers if split else block_q
    q_steps, stages = queries // block_q, min(program.tiles.stages, queries // block_q)
    graded, buffers, groups = _graded(kinds, shares, query.shape, key.shape[1])
    # A tile parameter's entries keys first, as the program's tiles hold them (`_param`).
    scaled_in = None if program.tiles.staged_logits else head_layout(family, scaled)
    leaves, treedef = jax.tree.flatten((spans, _keys_last(score_params, kinds), drop, scaled_in))
    tiled = Kind.TILE in graded

    def body(*refs, streams, after=None):
        inputs, unflattened, outputs, held, (key_refs, tile_refs) = _dkv_refs(
            refs, treedef, len(leaves), graded, groups, widths=value_width == width, split=split,
            fused=walk.fused, after=after is not None, vectors=family.stages_vectors)
        spans_of, score_refs, drop_of, l_ref = unflattened
        dq_ref, dk_ref, dv_ref = outputs
        q_ref, k_ref, v_ref, do_ref, *stat_refs = inputs
        k_smem, v_smem, out_smem, dv_smem, rows_smem, k_loaded, v_loaded, *rest = held
        relay, tile_stage, fused_stages, logits_stages = _held_stages(
            rest, relayed=family.relay_bytes(block_k, count) > 0, tiled=tiled,
            fused=(4 if placing else 2) if walk.fused else 0)
        row, head = family.program_index("batch"), family.program_index("heads")
        named = {"queries": q_ref, "douts": do_ref, **({} if l_ref is None else {"logits": l_ref})}
        views, streams = _query_streams(family, streams, named, (score_refs, kinds), row, head, heads=heads)
        pair_first = family.program_index("k_tiles") * pair
        streamed = _streamed([program.readers(pair_first + index * block_k, q_steps, spans_of, row)[:2]
                              for index in range(holders(family, consumers))])
        rows_of = functools.partial(_statistics_of, family, stat_refs, (row, head))
        ring = _PlacedDq(walk, dq_ref, tuple(fused_stages), (row, head)) if placing and walk.fused else None
        def consumer(index):
            holder = 0 if split or placing else index
            k_first = pair_first + holder * block_k
            # Shared between the consumers (`split_queries`, a placed walk's): the first loads them.
            _block_loaded(family, (k_ref, v_ref), (k_smem, v_smem), (k_loaded, v_loaded),
                          (row, lax.div(head, heads // kv_heads)), k_first, block_k, holder,
                          issued=index == 0 if split or placing else True)
            # Consumers that meet each step walk every streamed step in step, each masking the queries its own
            # keys do not see: a staged logits tile (one consumer writes it, every one reads it) takes them
            # all.
            lo, hi = (streamed if program.tiles.staged_logits
                      else program.readers(k_first, q_steps, spans_of, row)[:2])
            part = (index * count, count) if split else (0, block_q)
            stage = _logits_stage(family, consumers, logits_stages, index)
            step = _consumer_step(_relayed(walk, relay, index), (row, head), k_first, index, holder, part,
                                  stage, shares,
                                  refs=(tile_refs, tile_stage, spans_of, drop_of, score_refs, k_smem, v_smem,
                                        dq_ref, fused_stages),
                                  statistics=rows_of)
            # Each consumer's walk over the key block's readers (`Family.walk_own`).
            run = program.whole_run(k_first, q_steps, spans_of)
            key_sums, (dk, dv) = family.walk_own(
                streams, views, (lo, hi), streamed, index,
                lambda streams, lo, hi, counted: family.walk(
                    streams, program.segmented(step, run, lo, hi),
                    tuple(family.zeros((block_k,)) for _ in key_refs),
                    tuple((block_k, share) for _, share in shares), walk.interpret, first=counted))
            if ring is not None:
                ring.drained(lo, hi, index)
            if stage is not None:
                # The read turn's last phase, past the walk's last product, observed.
                family.turn(stage[2], consumers, arrive=False)
            if split or (placing and index):
                return (dk, dv, *key_sums) if split else None
            span = pl.ds(k_first, block_k)
            outs = tuple(ref.at[head_slice(family, row, head, span, ref.dtype, width=val.shape[1])]
                         for ref, val in ((dk_ref, dk), (dv_ref, dv)))
            key_views = tuple(ref.at[row, head, span] for ref in key_refs)
            _keys_stored(family, (dk, dv, *key_sums), (*outs, *key_views),
                         (out_smem, dv_smem, rows_smem), after, index=index, widths=value_width == width,
                         interpret=walk.interpret)
            return None
        reached = functools.partial(program.readers, pair_first, q_steps, spans_of, row)
        return scheduled_roles(family, consumers, consumer, functools.partial(
            family.stream, streams, views, *streamed), functools.partial(
            _keys_first_issued, walk, streams, reached, first=streamed[0],
            resident=(k_smem, v_smem, k_loaded, v_loaded), staged=fused_stages, shares=shares),
            reducer=None if ring is None else lambda role: ring.reduced(*reached()[:2], role))

    scoped, rings, after = _dkv_allocations(family, program.tiles, width, value_width, query.dtype,
                                            stages=stages, columns=program.tiles.columns, tiled=tiled,
                                            params=param_types(map(Param, score_params, kinds)))
    products = placed_scratch(family, block_q, query.dtype, dq_halves(width) if walk.fused else (),
                              tuple(share for _, share in shares))
    def entry(*refs):
        summed = family.scoped(functools.partial(body, *refs), *scoped, streams=rings,
                               after_walks=tuple(after), **products)
        if summed is not None:
            first = 9 + len(leaves) + walk.fused
            grads = stacked.views(refs[first:first + len(groups)], groups)
            outputs = refs[first + len(groups):first + len(groups) + 2]
            _summed_stored(family, summed, _shared_views(family, outputs, grads, graded, block_k), consumers,
                           walk.interpret)
    launch = _dkv_kernel(walk, entry, graded, query, key, value, dout, shares, name)
    query, key, value, dout = (head_layout(family, arr) for arr in (query, key, value, dout))
    grads = _dkv_launched(walk, launch, (query, key, value, dout, peak, total, anchor, delta, key_index,
                                         *leaves), graded, buffers, groups,
                          dq_shape=(query.shape[0], queries, heads, width))
    swapped = 3 if walk.fused else 2
    return (*(head_unlayout(family, grad, heads) for grad in grads[:swapped]), *grads[swapped:])

def _held_stages(rest: list, *, relayed: bool, tiled: bool, fused: int) -> tuple:
    """A dK/dV program's stages past its resident blocks and output stages, as `_dkv_allocations` lays them:
    its consumers' relay tiles where `relayed` and its tile gradients' stage where `tiled` (the last), a
    fused launch's `fused` stages and barriers (its placed walk's dQ ring, `_PlacedDq`), then its logits'."""
    relay, rest = (rest[-1:], rest[:-1]) if relayed else ([], rest)
    tile_stage, rest = (rest[-1], rest[:-1]) if tiled else (None, rest)
    return relay, tile_stage, rest[:fused], rest[fused:]


def _block_loaded(family: mosaic.Family, refs: tuple, smem: tuple, loaded: tuple, place: tuple,
                  k_first, block_k: int, holder, *, issued) -> None:
    """A holder's resident key block, its keys and values (`refs`) from `k_first` at its batch row and key
    head (`place`) copied into its `smem` (`Family.loaded`, each awaited on its `loaded` barrier), the
    copies issued where `issued` (one consumer's for every consumer that shares them)."""
    row, kv_head = place
    family.loaded(tuple(
        (ref.at[head_slice(family, row, kv_head, pl.ds(k_first, block_k), ref.dtype, width=tile.shape[-1])],
         tile.at[holder], barrier.at[holder])
        for ref, tile, barrier in zip(refs, smem, loaded, strict=True)), issued=issued)


def _placed_step_of(walk: _Backward, rows_of, spans_of, place: tuple, k_first, dq_ref, fused_stages
                    ) -> Callable:
    """A compute warpgroup's step of a walk placed on a role keys first (`_placed_step`) over the keys from
    `k_first` at its batch row and head (`place`), a fused walk's dQ added through its stages
    (`_PlacedDq`)."""
    row, head = place
    coords_of = functools.partial(_coords, walk.program.family, spans_of, row, k_first=k_first,
                                  keys_first=True, draw=None)
    added = _PlacedDq(walk, dq_ref, tuple(fused_stages), (row, head)) if walk.fused else None
    return functools.partial(_placed_step, walk, rows_of, coords_of, added)


def _keys_first_issued(walk: _Backward, streams, reached: Callable, walked: placed.Tcgen05Walked, *, first,
                       resident: tuple, staged, shares: tuple) -> None:
    """A walk's issuer where it is placed keys first (`placed.Tcgen05Walked.issuing`): its products
    (`KEYS_FIRST`, a fused walk's dQ too, `FUSED_DQ`) over the key block's readers (`reached()`'s first
    two, formed in the issuer's role), a step ahead of the compute warpgroups that read them, its resident
    keys and values (`resident`, their tiles and their loads' barriers) their lefts once in, a fused walk's
    `staged` score cotangents, dK's and dV's `shares` of their columns."""
    k_smem, v_smem, k_loaded, v_loaded = resident
    lefts = {"keys": k_smem.at[0], "values": v_smem.at[0], **({"staged": staged[0]} if walk.fused else {})}
    walked.issuing(streams, KEYS_FIRST + (FUSED_DQ if walk.fused else ()), *reached()[:2], first=first,
                   lefts=(lefts,), ready=(k_loaded.at[0], v_loaded.at[0]),
                   shares={placed.DK: shares[0], placed.DV: shares[1]})


def _query_streams(family: mosaic.Family, streams, named: dict, score: tuple, row, head, *,
                   heads: int) -> tuple[dict, dict]:
    """A dK/dV program's views of its query streams (`named`'s refs, `_views_of`) beside those of its score
    hook's parameters that ride them (`score`, their refs and kinds: `score_views`, at the program's key
    tile), and its streams opened with them."""
    refs, kinds = score
    views = _views_of(family, named, row, head, heads=heads)
    if riding(family, kinds, "queries"):
        views |= score_views(family, refs, kinds, "queries", row, head, family.program_index("k_tiles"))
    return views, family.opened(streams, views)


def _consumer_step(walk: _Backward, place: tuple, k_first, index: int, holder: int, part: tuple, stage,
                   shares: tuple, *, refs: tuple, statistics) -> Callable:
    """A dK/dV consumer's step over the keys from `k_first` (`_dkv_step`), its query `part` of a slot, its
    hook's coordinates, parameters and tile gradients' stores, its resident key and value block (its
    `holder`'s), a fused launch's dQ adds and its logits' `stage`, for its batch row and head (`place`); a
    compute warpgroup's where its family places its products on a role (`_placed_step_of`)."""
    tile_refs, tile_stage, spans_of, drop_of, score_refs, k_smem, v_smem, dq_ref, fused_stages = refs
    family = walk.program.family
    if family.places_on_role:
        return _placed_step_of(walk, statistics, spans_of, place, k_first, dq_ref, fused_stages)
    row, head = place
    store_tiles = functools.partial(_tiles_added, family, tile_refs, tile_stage, place, k_first, index,
                                    walk.interpret)
    coords_of = functools.partial(_coords, family, spans_of, row, k_first=k_first, keys_first=True,
                                  draw=None if drop_of is None else drop_of.draw(row, head))
    # The key block's parameters held across the walk, read once (`held_params`), each query tile's of the
    # queries' stream its step's, the consumer's queries of the slot (`_score_params`).
    block_k = walk.program.tiles.block_k
    params_of = functools.partial(
        _score_params, family, score_refs, walk.kinds, row, head, k_first=k_first, keys_first=True,
        held=held_params(family, score_refs, walk.kinds, row, head, k_first, block_k, along="queries"),
        streamed=riding(family, walk.kinds, "queries"), at=(0, part[0]))
    adds = _fused_adds(walk, dq_ref, fused_stages, place, index, k_smem, holder)
    return functools.partial(_dkv_step, walk, (k_smem.at[holder], v_smem.at[holder]), statistics, coords_of,
                             params_of, store_tiles, shares, part, stage, adds)


def _statistics_of(family: mosaic.Family, stat_refs: list, place: tuple, q_first, rows: int) -> tuple:
    """`rows` queries' statistics from `q_first` of a batch row and head (`place`) in a column's layout, the
    tiles' queries its columns: each a reader the scores call where they use it (`_scores`' `read`). Loaded
    at once ahead of the score hook's inputs, an sm86's and an
    sm89's dK/dV at width 64 spilled 80 bytes under a row-indexed hook and 8 under a tile bias, where an
    sm80's spilled none with the same geometry and PTX; read where used, none spills on any of the three
    (row 5467, ptxas 12.9.86)."""
    return tuple(functools.partial(family.columns_of, ref.at[*place, pl.ds(q_first, rows)])
                 for ref in stat_refs)


def _dkv_launched(walk: _Backward, launch, operands: tuple, graded: list[Kind], buffers: list,
                  groups: tuple, *, dq_shape: tuple[int, int, int, int]) -> tuple[jax.Array, ...]:
    """The dK/dV launch `launch` over `operands`: dK, dV and the hook's key and tile gradients in kind order,
    each into zeros at the call's batch rows and heads (`buffers`, `_grad_buffers`) whatever rows its
    parameter shares, added by the program that reads its place, the tiles' keys first `[., ., Kp, Qp]`; the
    shared rows summed after, each at its own scale (`template.hook_gradient`), where summing them in the
    program (row 2577) took them at one scale, each row's a power of two apart (`value_shift`); a fused
    one's dQ before them, added into zeros too, one copy, or each consumer's own interpreted
    (`Family.added`). The gradients of one shape (`groups`) are one ref, a leading axis of them
    (`stacked.zeros`, row 3180)."""
    grads = stacked.zeros(buffers, groups, *operands[:4])
    if not walk.fused:
        dk, dv = launch(*operands, *grads)
        return dk, dv, *_keys_last(stacked.read(grads, groups), graded)
    batch, queries, heads, width = dq_shape
    copies = walk.consumers if walk.interpret else 1
    dq_axes = head_slice(walk.program.family, batch, heads, queries, jnp.float32)
    dq = jax.new_ref(launches.filled((copies, *dq_axes, width), jnp.float32, *operands[:4]))
    dk, dv = launch(*operands, dq, *grads)
    return stacked.added(dq), dk, dv
