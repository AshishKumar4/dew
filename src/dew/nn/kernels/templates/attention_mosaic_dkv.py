"""The attention template's Mosaic GPU dK/dV launch keys first (`attention_mosaic_backward`): each consumer's
key block resident, walked by the query tiles that read it, its dK and dV, and the hook's key and tile
gradients; the fused fast form's keys-first launch adds each tile's dQ too."""

import functools
from collections.abc import Callable
from types import MappingProxyType

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from dew._devices import deterministic_ops_requested
from dew.nn.kernels import tuning
from dew.nn.kernels.templates import (
    across,
    attention_algebra as algebra,
    attention_gradient as gradient,
    mosaic,
    swizzling,
)
from dew.nn.kernels.templates.attention import DKV_GRADS, Dropout, Kind, by_power, shared_place
from dew.nn.kernels.templates.attention_mosaic_program import (
    BackwardTiles,
    TileWalk,
    _coords,
    _Spans,
    _streamed,
    _views_of,
    holders,
    issued_into,
    modelled,
    reducing,
    split_queries,
    stages_after_walks,
    stated,
    transposed_swizzles,
)
from dew.nn.kernels.templates.attention_mosaic_scores import (
    _Backward,
    _dq_added,
    _relay_tiles,
    _relayed,
    _score_params,
    _scores,
)


def _dkv_allocations(family: mosaic.Family, config: BackwardTiles, width: int, value_width: int, dtype, *,
                     stages: int, columns: int = 1, tiled: bool = False) -> tuple[list, dict, list]:
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
    are scoped after the walks where the family scopes them apart (`stages_after_walks`), in their slots."""
    consumers, block_q, block_k = config.consumers, config.block_q, config.block_k
    split = split_queries(family)
    holders, rows = (1, block_q // consumers) if split else (consumers, block_q)
    swizzle = swizzling.swizzle_for(dtype, rows, width, value_width) if split else None
    # Each holder's output stages, where it stores its own sums: the consumers sharing a key block store
    # theirs summed once, in a scope of their own after the walk's (`_summed_stored`).
    out_width = width // columns
    outputs = [] if split else [
        family.staged_memory((holders, max(block_q, block_k), out_width), jnp.float32),
        *([] if value_width == width
          else [family.staged_memory((holders, block_k, value_width // columns), jnp.float32)]),
        family.staged_memory((holders, block_k), jnp.float32)]
    after = outputs if stages_after_walks(family) else []
    scoped = [family.staged_memory((holders, block_k, width), dtype, swizzle),
              family.staged_memory((holders, block_k, value_width), dtype, swizzle),
              *([] if after else outputs), family.load_barriers(holders), family.load_barriers(holders)]
    blocks: dict[str, tuple] = {"queries": ((block_q, width), dtype),
                                "douts": ((block_q, value_width), dtype)}
    if config.fused:
        # A fused launch's (`_dkv_step`'s `add_dq`): each consumer's score cotangents staged for dQ's
        # product, their transpose its left operand, and its dQ tile's stage for the add.
        scoped += [family.staged_memory((consumers, block_k, rows), dtype),
                   family.staged_memory((consumers, rows, width), jnp.float32)]
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
    return scoped, family.streams(blocks, stages, consumers, swizzles=swizzles), after


def _dkv_step(walk: _Backward, resident: tuple, rows_of, coords_of, params_of, store_tiles, shares, part,
              logits_stage, add_dq, index, slots, carry, accumulators):
    """A dK/dV consumer's query tile `index`, its keys the tile's rows (`[bk, bq]`, FlashAttention 3's order,
    hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp's SdP_swapAB): its scores `K Qᵀ` (the call's power taken after
    the product) and score cotangents from the query and output cotangent slots (`slots`) and its resident
    keys and values (`resident`), under the tile's queries' statistics (`rows_of(first query)`, in a column's
    layout), its coordinates (`coords_of(first query)`) and its hook parameters (`params_of(first query,
    like)`). The weights and the score cotangents, their keys as their rows, are dV's and dK's left operands
    from the registers (`algebra.value_operands`, `cotangent_product`'s `aᵀ @ b`), issued into `accumulators`
    in place. The key gradients' sums (`carry`) take the pulled cotangents' row sums, and the tile gradients
    go to `store_tiles(first query, pieces)`, `[bk, bq]` each. The consumer takes `part` of each slot's
    queries (`(first, count)`, `split_queries`; the whole slot where its keys are its own)."""
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
    s = family.product(k_smem, logits_slot.transpose((1, 0)), shape)
    if logits_stage is not None:
        # Its share of the staged queries read (the product waited): the next step's stage may be written.
        walk.program.family.turn(logits_stage[2], walk.consumers, wait=False)
    dp = family.product(v_smem, do_slot.transpose((1, 0)), shape)
    # Every tile masked, its queries' statistics read past its products: a tile chosen masked or whole at
    # run time joins both choices' tiles and its column statistics, 16 registers each, past a consumer's
    # (spilled 1184 bytes at 64 x 64, one consumer).
    read = walk.places_read(masked=True)
    fused = add_dq is not None
    coords = coords_of(q_first, shape=shape, read=read)
    narrowed, logits, pieces = _weighed(walk, coords, params_of(q_first, s), s, dp, rows_of(q_first, count),
                                        (do_slot.dtype, q_slot.dtype), one_part=fused)
    # dK's and dV's products against the program's share of the queries' and output cotangents' columns.
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

def _weighed(walk: _Backward, coords: algebra.Coords, params: tuple, s, dp, rows: tuple, dtypes: tuple, *,
             one_part: bool) -> tuple:
    """A keys-first tile's: its weights narrowed for dV in the output cotangents' dtype
    (`algebra.value_weights`, the dropped ones, `algebra.drawn`; the dQ program's sums take them
    undropped), its pulled score cotangents and its hook's pieces (`gradient.pulled`), from its products `s`
    and `dp` under its queries' statistics `rows`, its coordinates and its hook's parameters (`_scores`);
    the cotangents the fast form's one part in the queries' dtype where the launch takes it from the
    registers (`one_part`, the fused launch's), narrowed as they are formed."""
    values, queries = dtypes
    exponentials, weights, cotangents, slopes, seen = _scores(walk, coords, params, s, dp, *rows, masked=True)
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

KEY_ROWS = MappingProxyType({Kind.KEY: lambda entries: entries.sum(axis=-1)})
"""`gradient.pulled`'s sums of a dK/dV tile, its keys its rows (`_dkv_step`): a key's over its row; the dK/dV
program writes no head's or query's gradient (`DKV_GRADS`)."""


def _tiles_added(family: mosaic.Family, refs, stage, place: tuple, k_first, index, interpret: bool, first,
                 pieces) -> None:
    """Each tile gradient of `pieces` (`[bk, its queries]`, queries from `first`) added into its place in the
    keys-first `[B or 1, H or 1, Kp, Qp]` refs (`refs`), the one a parameter the batch rows or heads share
    holds for the program's row and head (`place`, `shared_place`, row 2577), through consumer `index`'s
    `stage`."""
    row, head = place
    for ref, piece in zip(refs, pieces, strict=True):
        assert stage is not None
        view = ref.at[*shared_place(ref.shape, row, head), pl.ds(k_first, piece.shape[0]),
                      pl.ds(first, piece.shape[1])]
        family.added(piece, view, interpret, smem=stage.at[index])


def _dkv_stored(family: mosaic.Family, sums: tuple, views: tuple, stages: tuple, *, index, interpret: bool
                ) -> None:
    """A dK/dV consumer's dK and dV of its own keys to their `views`, and its key gradients' sums added into
    theirs (`Family.added`), or written to its own place where its family adds no vector
    (`Family.adds_vectors`, `_dkv_launched`), through its `stages` (dK's output stage, dV's, the key
    gradients')."""
    out_smem, dv_smem, rows_smem = stages
    dk, dv, *key_sums = sums
    dk_view, dv_view, *key_views = views
    dk_stage = out_smem.at[index, pl.ds(0, dk.shape[0])]
    if dv_smem is out_smem:
        family.stored((dk, dv), (dk_view, dv_view), smem=dk_stage)
    else:
        family.stored((dk,), (dk_view,), smem=dk_stage)
        family.stored((dv,), (dv_view,), smem=dv_smem.at[index])
    for key_sum, key_view in zip(key_sums, key_views, strict=True):
        # Each key gradient, the walk's row sums, added into its own `[B or 1, H or 1, Kp]`.
        if family.adds_vectors:
            family.added(key_sum, key_view, interpret, smem=rows_smem.at[index])
        else:
            family.stored((family.as_row(key_sum),), (key_view,), smem=rows_smem.at[index])

def _grad_buffers(graded: list[Kind], shapes: list, batch: int, heads: int, queries: int, keys_padded: int, *,
                  adds_vectors: bool) -> list[tuple]:
    """Each graded parameter's gradient buffer's shape (`_dkv_launched`): its own batch rows and heads
    (`shapes`, `[B or 1, H or 1, ...]`), or the call's in deterministic mode and for a key's where the
    launch's family adds no vector (`Family.adds_vectors`), then its keys, and for a tile its queries."""
    if deterministic_ops_requested():
        shapes = [(batch, heads)] * len(shapes)
    elif not adds_vectors:
        shapes = [(batch, heads) if kind == Kind.KEY else shape
                  for shape, kind in zip(shapes, graded, strict=True)]
    return [(*shape[:2], keys_padded) if kind == Kind.KEY else (*shape[:2], keys_padded, queries)
            for shape, kind in zip(shapes, graded, strict=True)]

def _graded(score_params: tuple, kinds: tuple, shares: tuple, shape: tuple, keys_padded: int, *,
            adds_vectors: bool) -> tuple[list[Kind], list[tuple], tuple[tuple[int, ...], ...]]:
    """The kinds of the hook gradients a dK/dV program of `shares` writes (`DKV_GRADS`; none for a later
    share's), their buffers' shapes (`_grad_buffers`) and the groups of one shape (`mosaic.stacked_groups`),
    for a call whose queries are `shape` `[B, Q, H, D]`."""
    graded = [kind for kind in kinds if kind in DKV_GRADS] if _grading(shares) else []
    graded_shapes = [param.shape for param, kind in zip(score_params, kinds, strict=True) if kind in graded]
    buffers = _grad_buffers(graded, graded_shapes, shape[0], shape[2], shape[1], keys_padded,
                            adds_vectors=adds_vectors)
    return graded, buffers, mosaic.stacked_groups(buffers)


def _dkv_refs(refs, treedef, leaves: int, graded: list[Kind], groups: tuple, *, widths: bool, split: bool,
              fused: bool, after: bool) -> tuple:
    """The dK/dV program's refs by role: its nine inputs, its spans' and hook parameters' (`treedef` of
    `leaves`), a fused launch's dQ and the hook's key and tile gradients, each added into in place, a ref a
    group of equal shapes (`_dkv_launched`, `mosaic.stacked_groups`), its dK and dV outputs, and its scoped
    allocations (`_dkv_scoped`)."""
    inputs, refs = refs[:9], refs[9:]
    unflattened, refs = treedef.unflatten(refs[:leaves]), refs[leaves:]
    dq_ref, refs = (refs[0], refs[1:]) if fused else (None, refs)
    grad_refs, refs = refs[:len(groups)], refs[len(groups):]
    views = mosaic.stacked_views(grad_refs, groups)
    dk_ref, dv_ref, *refs = refs
    keyed = tuple([ref for ref, kind in zip(views, graded, strict=True) if kind == wanted]
                  for wanted in DKV_GRADS)
    held = _dkv_scoped(refs, widths=widths, staged=not split and not after)
    return inputs, unflattened, (dq_ref, dk_ref, dv_ref), held, keyed

def _shared_views(family: mosaic.Family, outputs, grad_refs, graded: list[Kind], block_k: int) -> tuple:
    """dK's and dV's views (`outputs`, the dK/dV launch's) and each key gradient's (`grad_refs`, at the place
    it holds for the program's row and head, `shared_place`) at the program's key block, the one its
    consumers share (`split_queries`)."""
    dk_ref, dv_ref = outputs
    row, head = family.program_index("batch"), family.program_index("heads")
    keys = pl.ds(family.program_index("k_tiles") * block_k, block_k)
    key_views = [ref.at[*shared_place(ref.shape, row, head), keys]
                 for ref, kind in zip(grad_refs, graded, strict=True) if kind == Kind.KEY]
    return dk_ref.at[row, keys, head], dv_ref.at[row, keys, head], *key_views

def _summed_stored(family: mosaic.Family, sums: tuple, views: tuple, consumers: int, interpret: bool) -> None:
    """The sums of the `consumers` sharing a key block (each over its share of the queries) added in a fixed
    order and stored once to their `views` (`across.summed_across`), dK's and dV's, the key gradients' added
    into theirs (`Family.added`: a parameter the batch rows or heads share takes each program's), each in a
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

def _dkv_scoped(refs, *, widths: bool, staged: bool) -> tuple:
    """The dK/dV program's allocations (`_dkv_allocations`) by role: its resident keys and values, dK's output
    stage and dV's (dK's where the head `widths` agree) and its key gradients' stage where the program's
    scope holds them (`staged`: none where its consumers share its keys, `split_queries`, or a scope after
    the walks holds them, `_output_stages`), its two loads' barriers and its logits' stage where it has
    one."""
    keys, values, refs = refs[0], refs[1], refs[2:]
    if not staged:
        return keys, values, None, None, None, *refs
    dk_stage, dv_stage, rows_stage = _output_stages(refs[:3], widths=widths)
    return keys, values, dk_stage, dv_stage, rows_stage, *refs[3 - widths:]


def _output_stages(refs, *, widths: bool) -> tuple:
    """dK's output stage, dV's (dK's where the head `widths` agree) and the key gradients' from the first of
    `refs` (`_dkv_allocations`' outputs)."""
    return (refs[0], refs[0], refs[1]) if widths else (refs[0], refs[1], refs[2])

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
    dk, dv = (jax.ShapeDtypeStruct((batch, keys_padded, heads, share), jnp.float32) for _, share in shares)
    return family.kernel(
        entry, out_type=(dk, dv),
        grid=(heads, keys_padded // (holders(family, consumers) * tiles.block_k), batch),
        grid_names=("heads", "k_tiles", "batch"), consumers=consumers, interpret=walk.interpret,
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
        after(lambda *scoped: stored(_output_stages(scoped, widths=widths)))


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
    program, consumers, kinds = walk.program, walk.consumers, walk.kinds
    family = program.family
    block_q, block_k = program.tiles.block_q, program.tiles.block_k
    # The consumers share one key block, each its share of a slot's queries, where they feed their rings.
    split = split_queries(family)
    pair, count = (block_k, block_q // consumers) if split else (consumers * block_k, block_q)
    q_steps = queries // block_q
    stages = min(program.tiles.stages, q_steps)
    graded, buffers, groups = _graded(score_params, kinds, shares, query.shape, key.shape[1],
                                      adds_vectors=family.adds_vectors)
    # A tile parameter's entries keys first, as the program's tiles hold them (`_param`).
    staged_logits = program.tiles.staged_logits
    leaves, treedef = jax.tree.flatten((spans, _keys_last(score_params, kinds), drop,
                                        None if staged_logits else scaled))

    tiled = Kind.TILE in graded

    def body(*refs, streams, after=None):
        inputs, unflattened, outputs, held, (key_refs, tile_refs) = _dkv_refs(
            refs, treedef, len(leaves), graded, groups, widths=value_width == width, split=split,
            fused=walk.fused, after=after is not None)
        spans_of, score_refs, drop_of, l_ref = unflattened
        dq_ref, dk_ref, dv_ref = outputs
        q_ref, k_ref, v_ref, do_ref, *stat_refs = inputs
        k_smem, v_smem, out_smem, dv_smem, rows_smem, k_loaded, v_loaded, *rest = held
        relayed = family.relay_bytes(block_k, count) > 0
        relay, rest = (rest[-1:], rest[:-1]) if relayed else ([], rest)
        tile_stage, rest = (rest[-1], rest[:-1]) if tiled else (None, rest)
        tiles = program.tiles
        fused = 2 if walk.fused else 0
        fused_stages, logits_stages = rest[:fused], rest[fused:]
        row, head = family.program_index("batch"), family.program_index("heads")

        named = {"queries": q_ref, "douts": do_ref, **({} if l_ref is None else {"logits": l_ref})}
        views = _views_of(named, row, head)
        streams = family.opened(streams, views)
        pair_first = family.program_index("k_tiles") * pair
        holders = 1 if split else consumers
        streamed = _streamed([program.readers(pair_first + index * block_k, q_steps, spans_of, row)[:2]
                              for index in range(holders)])

        rows_of = functools.partial(_statistics_of, family, stat_refs, (row, head))

        def consumer(index):
            holder = 0 if split else index
            k_first = pair_first + holder * block_k
            keys_view = (row, pl.ds(k_first, block_k), lax.div(head, heads // kv_heads))
            copies = ((k_ref.at[keys_view], k_smem.at[holder], k_loaded.at[holder]),
                      (v_ref.at[keys_view], v_smem.at[holder], v_loaded.at[holder]))
            # Shared between the consumers (`split_queries`): the first loads them, each waits.
            family.loaded(copies, issued=index == 0 if split else True)
            # Consumers that meet each step walk every streamed step in step, each masking the queries its own
            # keys do not see: a staged logits tile (one consumer writes it, every one reads it) takes them
            # all.
            lo, hi = (streamed if tiles.staged_logits
                      else program.readers(k_first, q_steps, spans_of, row)[:2])
            part = (index * count, count) if split else (0, block_q)
            stage = _logits_stage(family, consumers, logits_stages, index)
            step = _consumer_step(_relayed(walk, relay, index), (row, head), k_first, index, holder, part,
                                  stage, shares,
                                  refs=(tile_refs, tile_stage, spans_of, drop_of, score_refs, k_smem, v_smem,
                                        dq_ref, fused_stages),
                                  statistics=rows_of)
            # The tiles its partner reads around its own, or its consumers' own loads of the query stream
            # where they feed it, the one key block's readers (`Family.walk_own`).
            key_sums, (dk, dv) = family.walk_own(
                streams, views, (lo, hi), streamed, index,
                lambda streams, lo, hi, counted: family.walk(
                    streams, ((step, lo, hi),), tuple(family.zeros((block_k,)) for _ in key_refs),
                    tuple((block_k, share) for _, share in shares), walk.interpret, first=counted))
            if stage is not None:
                # The read turn's last phase, past the walk's last product, observed.
                family.turn(stage[2], consumers, arrive=False)
            if split:
                # The consumers' sums, added in a scope after the walk's (`_summed_stored`).
                return dk, dv, *key_sums
            out = (row, pl.ds(k_first, block_k), head)
            key_views = tuple(ref.at[*shared_place(ref.shape, row, head), pl.ds(k_first, block_k)]
                              for ref in key_refs)
            _keys_stored(family, (dk, dv, *key_sums), (dk_ref.at[out], dv_ref.at[out], *key_views),
                         (out_smem, dv_smem, rows_smem), after, index=index, widths=value_width == width,
                         interpret=walk.interpret)
            return None

        return family.scheduled(consumers, consumer,
                                lambda: family.stream(streams, views, *streamed))

    scoped, rings, after = _dkv_allocations(family, program.tiles, width, value_width, query.dtype,
                                            stages=stages, columns=program.tiles.columns, tiled=tiled)

    def entry(*refs):
        summed = family.scoped(functools.partial(body, *refs), *scoped, streams=rings,
                               after_walks=tuple(after))
        if summed is not None:
            first = 9 + len(leaves) + walk.fused
            grads = mosaic.stacked_views(refs[first:first + len(groups)], groups)
            outputs = refs[first + len(groups):first + len(groups) + 2]
            _summed_stored(family, summed, _shared_views(family, outputs, grads, graded, block_k), consumers,
                           walk.interpret)

    launch = _dkv_kernel(walk, entry, graded, query, key, value, dout, shares, name)
    operands = (query, key, value, dout, peak, total, anchor, delta, key_index, *leaves)
    return _dkv_launched(walk, launch, operands, graded, buffers, groups)


def _consumer_step(walk: _Backward, place: tuple, k_first, index: int, holder: int, part: tuple, stage,
                   shares: tuple, *, refs: tuple, statistics) -> Callable:
    """A dK/dV consumer's step over the keys from `k_first` (`_dkv_step`), its query `part` of a slot, its
    hook's coordinates, parameters and tile gradients' stores, its resident key and value block (its
    `holder`'s), a fused launch's dQ adds and its logits' `stage`, for its batch row and head (`place`)."""
    tile_refs, tile_stage, spans_of, drop_of, score_refs, k_smem, v_smem, dq_ref, fused_stages = refs
    family = walk.program.family
    row, head = place
    store_tiles = functools.partial(_tiles_added, family, tile_refs, tile_stage, place, k_first, index,
                                    walk.interpret)
    coords_of = functools.partial(_coords, family, spans_of, row, k_first=k_first, keys_first=True,
                                  draw=None if drop_of is None else drop_of.draw(row, head))
    params_of = functools.partial(_score_params, family, score_refs, walk.kinds, row, head, k_first=k_first,
                                  keys_first=True)
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
                  groups: tuple) -> tuple[jax.Array, ...]:
    """The dK/dV launch `launch` over `operands`: dK, dV and the hook's key and tile gradients in kind order,
    each parameter's (`buffers`, its own batch rows and heads, `[B or 1, H or 1, ...]`) added into zeros in
    place by every program that reads it, the tiles' keys first `[., ., Kp, Qp]` (row 2577: a bias every row
    shares summed in the program, not written a row at a time); in deterministic mode each program's own
    place, the call's `[B, H, ...]`, written once, the shared rows summed in a fixed order after
    (`template.shared_sum`; row 2620: the adds of the programs sharing one place land in no fixed order); a
    fused one's dQ before them, added into zeros too, one copy, or each consumer's own interpreted
    (`Family.added`). A key gradient where the launch's family adds no vector (`Family.adds_vectors`) its
    program's own place too, written, the shared rows summed after (`_grad_buffers`). The gradients of one
    shape (`groups`) are one ref, a leading axis of them (`mosaic.stacked_zeros`, row 3180)."""
    batch, queries, heads, _ = operands[0].shape
    grads = mosaic.stacked_zeros(buffers, groups)
    if not walk.fused:
        dk, dv = launch(*operands, *grads)
        return dk, dv, *_keys_last(mosaic.stacked_read(grads, groups), graded)
    width = operands[0].shape[3]
    copies = walk.consumers if walk.interpret else 1
    dq = jax.new_ref(jnp.zeros((copies, batch, queries, heads, width), jnp.float32))
    dk, dv = launch(*operands, dq, *grads)
    return jax.freeze(dq).sum(axis=0), dk, dv
