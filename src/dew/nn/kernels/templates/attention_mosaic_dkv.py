"""The attention template's Mosaic GPU dK/dV launch keys first (`attention_mosaic_backward`): each consumer's
key block resident, walked by the query tiles that read it, its dK and dV, and the hook's key and tile
gradients; the fused fast form's keys-first launch adds each tile's dQ too."""

import functools
from types import MappingProxyType

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.devices import deterministic_ops_requested
from dew.nn.kernels.templates import attention_algebra as algebra, mosaic
from dew.nn.kernels.templates.attention import DKV_GRADS, Dropout, Kind, by_power, shared_place
from dew.nn.kernels.templates.attention_mosaic_program import (
    BackwardTiles,
    _coords,
    _Spans,
    _streamed,
    _views_of,
    described,
    kernel_of,
    reducing,
    stated,
)
from dew.nn.kernels.templates.attention_mosaic_scores import (
    _Backward,
    _dq_added,
    _score_params,
    _scores,
    split_queries,
)


def _dkv_allocations(family: mosaic.Family, config: BackwardTiles, width: int, value_width: int, dtype, *,
                     stages: int, columns: int = 1, tiled: bool = False) -> tuple[list, dict]:
    """The dK/dV program's allocations as `Family.scoped` takes them, in `_dkv_launch`'s order: its
    consumers' resident keys and values; dK's output stage and dV's (the one stage where the widths agree),
    fp32, each its program's share of the `columns`; each consumer's stage for its key gradients' sums; the
    two loads' barriers; a fused launch's (`BackwardTiles.fused`) staged score cotangents and dQ tile each
    consumer's; where its logits' queries are staged (`BackwardTiles.staged_logits`) one slot's of them and
    their two turns; where a score hook's tile parameter is `tiled`, each consumer's stage for its tile
    gradients' adds (`Family.added`); and its query and output-cotangent streams, and its logits' queries'
    unless staged, in `stages` slots (`Family.streams`). On a layout whose consumers feed their rings
    (`split_queries`) the consumers share one key block, held once, and each takes its share of a slot's
    queries: every block a product of theirs reads staged at the swizzle their share's contraction takes
    (`mosaic.swizzle_for`)."""
    consumers, block_q, block_k = config.consumers, config.block_q, config.block_k
    split = split_queries(family)
    holders, rows = (1, block_q // consumers) if split else (consumers, block_q)
    swizzle = mosaic.swizzle_for(dtype, rows, width, value_width) if split else None
    # Each holder's output stages, where it stores its own sums: the consumers sharing a key block store
    # theirs summed once, in a scope of their own after the walk's (`_summed_stored`).
    outputs = [] if split else [
        plgpu.SMEM((holders, max(block_q, block_k), width // columns), jnp.float32),
        *([] if value_width == width
          else [plgpu.SMEM((holders, block_k, value_width // columns), jnp.float32)]),
        plgpu.SMEM((holders, block_k), jnp.float32)]
    scoped = [mosaic.staged((holders, block_k, width), dtype, swizzle),
              mosaic.staged((holders, block_k, value_width), dtype, swizzle), *outputs,
              family.load_barriers(holders), family.load_barriers(holders)]
    blocks = {"queries": ((block_q, width), dtype), "douts": ((block_q, value_width), dtype)}
    if config.fused:
        # A fused launch's (`_dkv_step`'s `add_dq`): each consumer's score cotangents staged for dQ's
        # product, their transpose its left operand, and its dQ tile's stage for the add.
        scoped += [plgpu.SMEM((consumers, block_k, rows), dtype),
                   plgpu.SMEM((consumers, rows, width), jnp.float32)]
    if config.staged_logits:
        # One slot's queries taken by the power, which the first consumer stages for every consumer's share,
        # and the two turns that order its writes against their products' reads (`_staged_logits`).
        scoped += [mosaic.staged((block_q, width), dtype, swizzle), family.schedule(consumers),
                   family.schedule(consumers)]
    else:
        blocks["logits"] = ((block_q, width), dtype)
    if tiled:
        scoped.append(plgpu.SMEM((consumers, block_k, rows), jnp.float32))
    swizzles = None if swizzle is None else dict.fromkeys(blocks, swizzle)
    return scoped, family.streams(blocks, stages, consumers, swizzles=swizzles)

def _holders(family: mosaic.Family, consumers: int) -> int:
    """The key blocks a dK/dV program holds: one its consumers share (`split_queries`), else one each."""
    return 1 if split_queries(family) else consumers

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
    params = params_of(q_first, s)
    # Every tile masked, its queries' statistics read past its products: a tile chosen masked or whole at
    # run time joins both choices' tiles and its column statistics, 16 registers each, past a consumer's
    # (spilled 1184 bytes at 64 x 64, one consumer).
    coords = coords_of(q_first, shape=shape)
    exponentials, weights, cotangents, slopes, seen = _scores(
        walk, coords, params, s, dp, *rows_of(q_first, count), masked=True)
    logits, pieces = algebra.pulled(cotangents, slopes, seen, kinds, KEY_ROWS)
    # dV's weights the dropped ones (`algebra.value_operands`); the dQ program's sums take them undropped.
    multiplier = algebra.drawn(coords, coords.q_pos, coords.k_pos)
    if multiplier is not None:
        exponentials, weights = exponentials * multiplier, weights * multiplier
    narrowed = algebra.value_weights(exponentials, weights, do_slot.dtype, multiplier is not None)
    # dK's and dV's products against the program's share of the queries' and output cotangents' columns.
    (q0, q_share), (v0, v_share) = shares
    q_columns, do_columns = q_slot.at[:, pl.ds(q0, q_share)], do_slot.at[:, pl.ds(v0, v_share)]
    dv_acc = family.register_products(dv_acc, ((narrowed, None),), do_columns, walk.interpret)
    if add_dq is None:
        # Each part formed as its product is issued (`algebra.formed_parts`).
        parts = algebra.formed_parts(logits * walk.grad_scale, q_slot.dtype, accurate=walk.accurate)
        dk_acc = family.register_products(dk_acc, parts, q_columns, walk.interpret)
    else:
        # The fused launch's (the fast form's one part): dK's left operand from the registers, and staged,
        # its transpose dQ's (`algebra.cotangent_product`'s `a @ b`), the tile's dQ added into dQ, the
        # query tile's other key blocks adding theirs; a consumer's share of a 128-query slot is 64 rows,
        # warpgroup MMA's least (`MosaicBackward.legal`).
        ((part_tile, _),) = algebra.cotangent_parts(logits * walk.grad_scale, q_slot.dtype, accurate=False)
        dk_acc = family.register_products(dk_acc, ((part_tile, None),), q_columns, walk.interpret)
        staged_part, dq_stage = add_dq
        family.staged(part_tile, staged_part)
        dq_stage(q_first, family.product(staged_part.transpose((1, 0)), k_smem, (count, k_smem.shape[-1])))
    if _grading(shares):
        store_tiles(q_first, [piece for kind, piece in zip(kinds, pieces, strict=True) if kind == Kind.TILE])
        key_pieces = [piece for kind, piece in zip(kinds, pieces, strict=True) if kind == Kind.KEY]
        carry = tuple(total + piece for total, piece in zip(carry, key_pieces, strict=True))
    return carry, (dk_acc, dv_acc)

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

def _fused_adds(walk: _Backward, dq_ref, stages: list, place: tuple, index) -> tuple | None:
    """A fused launch's consumer `index`'s dQ adds (`_dkv_step`'s `add_dq`): its score cotangents' stage and
    its tiles' add into dQ at `place` (`_dq_added`); None where the launch adds none."""
    if not walk.fused:
        return None
    return stages[0].at[index], functools.partial(_dq_added, walk, dq_ref, stages[1], place, index)

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
"""`algebra.pulled`'s sums of a dK/dV tile, its keys its rows (`_dkv_step`): a key's over its row; the dK/dV
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


def _dkv_stored(family: mosaic.Family, sums: tuple, views: tuple, stages: tuple, index, interpret: bool
                ) -> None:
    """A dK/dV consumer's dK and dV of its own keys to their `views`, and its key gradients' sums added into
    theirs (`Family.added`), through its `stages` (dK's output stage, dV's, the key gradients')."""
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
        family.added(key_sum, key_view, interpret, smem=rows_smem.at[index])

def _dkv_refs(refs, treedef, leaves: int, graded: list[Kind], *, widths: bool, split: bool, fused: bool
              ) -> tuple:
    """The dK/dV program's refs by role: its nine inputs, its spans' and hook parameters' (`treedef` of
    `leaves`), a fused launch's dQ and the hook's key and tile gradients, each added into in place
    (`_dkv_launched`), its dK and dV outputs, and its scoped allocations (`_dkv_scoped`)."""
    inputs, refs = refs[:9], refs[9:]
    unflattened, refs = treedef.unflatten(refs[:leaves]), refs[leaves:]
    dq_ref, refs = (refs[0], refs[1:]) if fused else (None, refs)
    grad_refs, refs = refs[:len(graded)], refs[len(graded):]
    dk_ref, dv_ref, *refs = refs
    keyed = tuple([ref for ref, kind in zip(grad_refs, graded, strict=True) if kind == wanted]
                  for wanted in (Kind.KEY, Kind.TILE))
    return inputs, unflattened, (dq_ref, dk_ref, dv_ref), _dkv_scoped(refs, widths=widths, split=split), keyed

def _shared_views(outputs, grad_refs, graded: list[Kind], block_k: int) -> tuple:
    """dK's and dV's views (`outputs`, the dK/dV launch's) and each key gradient's (`grad_refs`, at the place
    it holds for the program's row and head, `shared_place`) at the program's key block, the one its
    consumers share (`split_queries`)."""
    dk_ref, dv_ref = outputs
    row, head = lax.axis_index("batch"), lax.axis_index("heads")
    keys = pl.ds(lax.axis_index("k_tiles") * block_k, block_k)
    key_views = [ref.at[*shared_place(ref.shape, row, head), keys]
                 for ref, kind in zip(grad_refs, graded, strict=True) if kind == Kind.KEY]
    return dk_ref.at[row, keys, head], dv_ref.at[row, keys, head], *key_views

def _summed_stored(family: mosaic.Family, sums: tuple, views: tuple, consumers: int, interpret: bool) -> None:
    """The sums of the `consumers` sharing a key block (each over its share of the queries) added in a fixed
    order and stored once to their `views` (`mosaic.summed_across`), dK's and dV's, the key gradients' added
    into theirs (`Family.added`: a parameter the batch rows or heads share takes each program's), each in a
    scope after the walk's, whose shared memory it reuses (`mosaic.summing`): the walk's last adds' reads of
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
            mosaic.summed_across(value, scratch, lax.axis_index("wg"), consumers,
                                 lambda summed: written(summed, stage))

        pl.run_scoped(total, plgpu.SMEM(value.shape, jnp.float32), **mosaic.summing(value.shape, consumers),
                      collective_axes="wg")

def _dkv_scoped(refs, *, widths: bool, split: bool) -> tuple:
    """The dK/dV program's allocations (`_dkv_allocations`) by role: its resident keys and values, dK's output
    stage and dV's (dK's where the head `widths` agree) and its key gradients' stage (none where its
    consumers share its keys, `split`), its two loads' barriers and its logits' stage where it has one."""
    keys, values, refs = refs[0], refs[1], refs[2:]
    if split:
        return keys, values, None, None, None, *refs
    dk_stage, refs = refs[0], refs[1:]
    dv_stage, refs = (dk_stage, refs) if widths else (refs[0], refs[1:])
    return keys, values, dk_stage, dv_stage, *refs

def _keys_last(arrays, kinds) -> tuple[jax.Array, ...]:
    """Each tile kind's of `arrays` (a score hook parameter's or gradient's, `[B, H, ., .]`) with its last two
    axes swapped, between the call's queries-first entries and the dK/dV program's keys-first ones."""
    return tuple(jnp.swapaxes(array, -1, -2) if kind == Kind.TILE else array
                 for array, kind in zip(arrays, kinds, strict=True))

def _dkv_kernel(walk: _Backward, entry, query, key, value, dout, shares: tuple, *, name: str,
                kernel_name: str | None):
    """The dK/dV program `entry`'s launch (`Family.kernel`) over `query` `[B, Qp, H, D]` and `key` and
    `value` `[B, Kp, Hk, D]`: dK and dV `[B, Kp, H, D]` fp32 out, each its share's columns (`shares`), the
    hook's key and tile gradients added into refs of their own (`_dkv_launched`); a program a query head's
    key block a consumer, or one its consumers share (`_holders`)."""
    batch, queries, heads, width = query.shape
    keys_padded, value_width = key.shape[1], value.shape[-1]
    family, consumers = walk.program.family, walk.consumers
    dk, dv = (jax.ShapeDtypeStruct((batch, keys_padded, heads, share), jnp.float32) for _, share in shares)
    return family.kernel(
        entry, out_type=(dk, dv),
        grid=(heads, keys_padded // (_holders(family, consumers) * walk.program.tiles.block_k), batch),
        grid_names=("heads", "k_tiles", "batch"), consumers=consumers, interpret=walk.interpret,
        approx_math=walk.approx_math,
        name=name, kernel_name=kernel_name,
        reducing_warpgroups=reducing(family, consumers, walk.spec.score_mod),
        # The model's products this launch owns (`stated`): its shares of dK's and dV's, dP's in the first
        # share's and dQ's where it adds it; its `S` (and a later share's dP) recomputed.
        cost=stated(walk.spec, batch, heads, walk.spec.queries or queries,
                    (width if walk.fused else 0) + shares[0][1] + shares[1][1]
                    + (value_width if shares[0][0] == 0 else 0), query, key, value, dout))

def _dkv_launch(query, key, value, dout, peak, total, anchor, delta, key_index, spans: _Spans | None,
                score_params: tuple, drop: Dropout | None, scaled: jax.Array, *, walk: _Backward
                ) -> tuple[jax.Array, ...]:
    """`_dkv_columns`' launch of each of the config's `columns`, its share of dK's and dV's columns (the
    score hook's key and tile gradients the first's), dK and dV joined along their widths; `scaled` the
    queries taken by the call's power, its logits' (`_dkv_step`)."""
    columns = walk.program.tiles.columns
    width, value_width = query.shape[-1] // columns, value.shape[-1] // columns
    launched = [_dkv_columns(query, key, value, dout, peak, total, anchor, delta, key_index, spans,
                             score_params, drop, scaled, walk=walk,
                             shares=((column * width, width), (column * value_width, value_width)))
                for column in range(columns)]
    if columns == 1:
        return launched[0]
    dk, dv = (jnp.concatenate([out[place] for out in launched], axis=-1) for place in (0, 1))
    return dk, dv, *launched[0][2:]

def _dkv_columns(query, key, value, dout, peak, total, anchor, delta, key_index, spans: _Spans | None,
                 score_params: tuple, drop: Dropout | None, scaled: jax.Array, *, walk: _Backward,
                 shares: tuple) -> tuple[jax.Array, ...]:
    """The dK/dV program under the dQ program's softmax: each consumer's `block_k` keys of a query head,
    resident, walked by the query tiles that see them, streamed with their output cotangents through a ring
    each; a tile's weights and score cotangents (the algebra's, rows the keys, `_dkv_step`) the transposed
    products' left operands from the registers (`algebra.value_operands`, `cotangent_product`). Returns the
    query head's dK and dV `[B, Kp, H, D]` fp32, then each key and tile gradient of the score hook in kind
    order (`DKV_GRADS`: keys `[B, H, Kp]`, tiles `[B, H, Qp, Kp]`, a tile parameter read and its gradient
    written keys first). A key or tile gradient is written by the one program that holds its keys; the
    tiles its walk does not read are zeros. Its dK and dV are its `shares` of their columns (`(first, count)`
    each); a later share's program writes no hook gradient."""
    kinds, width = walk.kinds, query.shape[-1]
    graded = [kind for kind in kinds if kind in DKV_GRADS] if _grading(shares) else []
    # A tile parameter's entries keys first, as the program's tiles hold them (`_param`).
    leaves, treedef = jax.tree.flatten((spans, _keys_last(score_params, kinds), drop,
                                        None if walk.program.tiles.staged_logits else scaled))
    name = (("attention_dkv_fused" if walk.fused else "attention_dkv")
            + ("" if shares[0][1] == width else f"_columns{shares[0][0] // shares[0][1]}"))
    launch = kernel_of(_dkv_built, name, walk, described((query, key, value, dout)), treedef, shares,
                       tuple(graded), hooks=(walk.spec.score_mod,))
    operands = (query, key, value, dout, peak, total, anchor, delta, key_index, *leaves)
    graded_params = [param for param, kind in zip(score_params, kinds, strict=True) if kind in graded]
    return _dkv_launched(walk, launch, operands, graded, [param.shape for param in graded_params],
                         key.shape[1])


def _dkv_built(walk: _Backward, operands: tuple, treedef, shares: tuple, grades: tuple[Kind, ...], *,
               name: str, kernel_name: str | None):
    """`_dkv_columns`' kernel for its static facts (`kernel_of`): the walk, its operands' shapes and dtypes,
    the structure of the arrays its programs read beside them, its shares of the columns and the hook
    gradients it writes (`grades`)."""
    query, key, value, dout = operands
    graded = list(grades)
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

    tiled = Kind.TILE in graded

    def body(*refs, streams):
        inputs, unflattened, outputs, held, (key_refs, tile_refs) = _dkv_refs(
            refs, treedef, treedef.num_leaves, graded, widths=value_width == width, split=split,
            fused=walk.fused)
        (spans_of, score_refs, drop_of, l_ref), (dq_ref, dk_ref, dv_ref) = unflattened, outputs
        q_ref, k_ref, v_ref, do_ref, *stat_refs = inputs
        k_smem, v_smem, out_smem, dv_smem, rows_smem, k_loaded, v_loaded, *rest = held
        tile_stage, rest = (rest[-1], rest[:-1]) if tiled else (None, rest)
        fused_stages, logits_stages = (rest[:2], rest[2:]) if walk.fused else ([], rest)
        row, head = lax.axis_index("batch"), lax.axis_index("heads")

        named = {"queries": q_ref, "douts": do_ref, **({} if l_ref is None else {"logits": l_ref})}
        views = _views_of(named, row, head)
        streams = family.opened(streams, views)
        pair_first = lax.axis_index("k_tiles") * pair
        holders = 1 if split else consumers
        streamed = _streamed([program.readers(pair_first + index * block_k, q_steps, spans_of, row)[:2]
                              for index in range(holders)])

        def rows_of(q_first, rows):
            # `rows` queries' statistics from `q_first` in a column's layout, the tiles' queries its columns.
            return tuple(family.columns_of(ref.at[row, head, pl.ds(q_first, rows)]) for ref in stat_refs)

        def consumer(index):
            holder = 0 if split else index
            k_first = pair_first + holder * block_k
            keys_view = (row, pl.ds(k_first, block_k), lax.div(head, heads // kv_heads))
            copies = ((k_ref.at[keys_view], k_smem.at[holder], k_loaded.at[holder]),
                      (v_ref.at[keys_view], v_smem.at[holder], v_loaded.at[holder]))
            # Shared between the consumers (`split_queries`): the first loads them, each waits.
            family.loaded(copies, issued=index == 0 if split else True)
            lo, hi, _ = program.readers(k_first, q_steps, spans_of, row)
            part = (index * count, count) if split else (0, block_q)

            store_tiles = functools.partial(_tiles_added, family, tile_refs, tile_stage, (row, head), k_first,
                                            index, walk.interpret)
            coords_of = functools.partial(_coords, family, spans_of, row, k_first=k_first, keys_first=True,
                                          draw=None if drop_of is None else drop_of.draw(row, head))
            params_of = functools.partial(_score_params, family, score_refs, kinds, row, head,
                                          k_first=k_first, keys_first=True)
            resident = k_smem.at[holder], v_smem.at[holder]
            adds = _fused_adds(walk, dq_ref, fused_stages, (row, head), index)
            stage = _logits_stage(family, consumers, logits_stages, index)
            step = functools.partial(_dkv_step, walk, resident, rows_of, coords_of, params_of, store_tiles,
                                     shares, part, stage, adds)
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
            _dkv_stored(family, (dk, dv, *key_sums), (dk_ref.at[out], dv_ref.at[out], *key_views),
                        (out_smem, dv_smem, rows_smem), index, walk.interpret)
            return None

        def producer():
            family.stream(streams, views, *streamed)

        return family.roles(consumers, consumer, producer)

    scoped, rings = _dkv_allocations(family, program.tiles, width, value_width, query.dtype, stages=stages,
                                     columns=program.tiles.columns, tiled=tiled)

    def entry(*refs):
        summed = family.scoped(functools.partial(body, *refs), *scoped, streams=rings)
        if summed is not None:
            first = 9 + treedef.num_leaves + walk.fused
            grads = refs[first:first + len(graded)]
            outputs = refs[first + len(graded):first + len(graded) + 2]
            _summed_stored(family, summed, _shared_views(outputs, grads, graded, block_k), consumers,
                           walk.interpret)

    return _dkv_kernel(walk, entry, query, key, value, dout, shares, name=name, kernel_name=kernel_name)

def _dkv_launched(walk: _Backward, launch, operands: tuple, graded: list[Kind], shapes: list,
                  keys_padded: int) -> tuple[jax.Array, ...]:
    """The dK/dV launch `launch` over `operands`: dK, dV and the hook's key and tile gradients in kind order,
    each parameter's (of `shapes`, its own batch rows and heads, `[B or 1, H or 1, ...]`) added into zeros in
    place by every program that reads it, the tiles' keys first `[., ., Kp, Qp]` (row 2577: a bias every row
    shares summed in the program, not written a row at a time); in deterministic mode each program's own
    place, the call's `[B, H, ...]`, written once, the shared rows summed in a fixed order after
    (`template.shared_sum`; row 2620: the adds of the programs sharing one place land in no fixed order); a
    fused one's dQ before them, added into zeros too, one copy, or each consumer's own interpreted
    (`Family.added`)."""
    batch, queries, heads, _ = operands[0].shape
    if deterministic_ops_requested():
        shapes = [(batch, heads)] * len(shapes)
    grads = [jax.new_ref(jnp.zeros((*shape[:2], keys_padded) if kind == Kind.KEY
                                   else (*shape[:2], keys_padded, queries), jnp.float32))
             for shape, kind in zip(shapes, graded, strict=True)]
    if not walk.fused:
        dk, dv = launch(*operands, *grads)
        return dk, dv, *_keys_last([jax.freeze(grad) for grad in grads], graded)
    width = operands[0].shape[3]
    copies = walk.consumers if walk.interpret else 1
    dq = jax.new_ref(jnp.zeros((copies, batch, queries, heads, width), jnp.float32))
    dk, dv = launch(*operands, dq, *grads)
    return jax.freeze(dq).sum(axis=0), dk, dv
