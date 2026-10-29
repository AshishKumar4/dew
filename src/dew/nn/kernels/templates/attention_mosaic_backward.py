"""The attention template's Mosaic GPU backward (`attention_mosaic.MosaicBackward`'s launches): D about each
query's anchor from a training forward's split sums (`algebra.split_delta`), then a dQ launch and a dK/dV
launch, `dq_finish`'s refinement between them, in either form, or the fast form's one launch, the dK/dV
program adding each tile's dQ (`_launch_backward`). Each launch's programs walk the bundle's streams as the
forward's do (`attention_mosaic_program`), the algebra's steps between its products."""

import dataclasses
import functools
from types import MappingProxyType

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import attention_algebra as algebra, bundles, mosaic
from dew.nn.kernels.templates.attention import (
    DKV_GRADS,
    DQ_GRADS,
    Dropout,
    Hooks,
    Kind,
    PaddedCall,
    Param,
    by_power,
)
from dew.nn.kernels.templates.attention_mosaic_program import (
    BackwardTiles,
    _columns,
    _coords,
    _param,
    _Program,
    _rows,
    _Spans,
    _spans_of,
    _streamed,
    _views_of,
    approximate,
    reducing,
    stated,
)
from dew.nn.precision import running_sum


def _dq_allocations(family: mosaic.Family, config: BackwardTiles, width: int, value_width: int, dtype,
                    stages: int) -> tuple[list, dict]:
    """The dQ program's allocations as `Family.scoped` takes them, in `_dq_launch`'s order: its consumers'
    query and output-cotangent stages, their dQ output stage (fp32, its share of the config's `columns`) and
    their rows' stage, the two loads' barriers, and its key and value streams in `stages` slots
    (`Family.streams`)."""
    consumers, block_q, block_k = config.consumers, config.block_q, config.block_k
    scoped = [plgpu.SMEM((consumers, block_q, width), dtype),
              plgpu.SMEM((consumers, block_q, value_width), dtype),
              plgpu.SMEM((consumers, block_q, width // config.columns), jnp.float32),
              plgpu.SMEM((consumers, block_q), jnp.float32),
              plgpu.Barrier(num_barriers=consumers), plgpu.Barrier(num_barriers=consumers)]
    blocks = {"keys": ((block_k, width), dtype), "values": ((block_k, value_width), dtype)}
    return scoped, family.streams(blocks, stages, consumers)


def _dkv_allocations(family: mosaic.Family, config: BackwardTiles, width: int, value_width: int, dtype, *,
                     stages: int, columns: int = 1) -> tuple[list, dict]:
    """The dK/dV program's allocations as `Family.scoped` takes them, in `_dkv_launch`'s order: its
    consumers' resident keys and values; dK's output stage and dV's (the one stage where the widths agree),
    fp32, each its program's share of the `columns`; each consumer's stage for its key gradients' sums; the
    two loads' barriers; a fused launch's (`BackwardTiles.fused`) staged score cotangents and dQ tile each
    consumer's; where its logits' queries are staged (`BackwardTiles.staged_logits`) each consumer's tile of
    them; and its query and output-cotangent streams, and its logits' queries' unless
    staged, in `stages` slots (`Family.streams`). On a layout whose consumers feed their rings
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
              plgpu.Barrier(num_barriers=holders), plgpu.Barrier(num_barriers=holders)]
    blocks = {"queries": ((block_q, width), dtype), "douts": ((block_q, value_width), dtype)}
    if config.fused:
        # A fused launch's (`_dkv_step`'s `add_dq`): each consumer's score cotangents staged for dQ's
        # product, their transpose its left operand, and its dQ tile's stage for the add.
        scoped += [plgpu.SMEM((consumers, block_k, rows), dtype),
                   plgpu.SMEM((consumers, rows, width), jnp.float32)]
    if config.staged_logits:
        scoped.append(mosaic.staged((consumers, rows, width), dtype, swizzle))
    else:
        blocks["logits"] = ((block_q, width), dtype)
    swizzles = None if swizzle is None else dict.fromkeys(blocks, swizzle)
    return scoped, family.streams(blocks, stages, consumers, swizzles=swizzles)


def _fused_allocations(family: mosaic.Family, config: BackwardTiles, width: int, dtype, stages: int
                       ) -> tuple[list, dict]:
    """The fused launch's allocations as `Family.scoped` takes them, in `_fused_launch`'s order: its
    consumers' resident keys and values; a tile's weights and score cotangents staged for the transposed
    products' left operands and dQ's; dK's and dV's output stage, which a tile's dQ adds take too (fp32, the
    head widths equal, `MosaicBackward.legal`); the two loads' barriers; and its query and output-cotangent
    streams in `stages` slots (`Family.streams`)."""
    consumers, block_q, block_k = config.consumers, config.block_q, config.block_k
    scoped = [plgpu.SMEM((consumers, block_k, width), dtype), plgpu.SMEM((consumers, block_k, width), dtype),
              plgpu.SMEM((consumers, 2, block_q, block_k), dtype),
              plgpu.SMEM((consumers, max(block_q, block_k), width), jnp.float32),
              plgpu.Barrier(num_barriers=consumers), plgpu.Barrier(num_barriers=consumers)]
    blocks = {"queries": ((block_q, width), dtype), "douts": ((block_q, width), dtype)}
    return scoped, family.streams(blocks, stages, consumers)


def _holders(family: mosaic.Family, consumers: int) -> int:
    """The key blocks a dK/dV program holds: one its consumers share (`split_queries`), else one each."""
    return 1 if split_queries(family) else consumers


def split_queries(family: mosaic.Family) -> bool:
    """Whether the dK/dV program's consumers share one key block and split each slot's queries between them:
    on a layout whose consumers feed their own rings (`mosaic.ConsumerFed`), each consumer's step its share
    of the queries' tiles and per-query statistics (`a05-twoconsumer`); else each consumer its own keys."""
    return isinstance(family, mosaic.ConsumerFed)


@dataclasses.dataclass(frozen=True)
class _Backward:
    """A backward launch's static choices: its program's geometry, its forward's (`spec`: the call's form, the
    logits' scale and units, its score hook and initial state), its form, its consumers, whether it is
    interpreted, the score hook's parameter kinds (`kinds`), and whether the dK/dV program adds each tile's
    dQ too (`fused`: the hookless fast form's one launch, dQ by atomic adds, D the forward's anchored one
    unrefined)."""

    program: _Program[BackwardTiles]
    spec: algebra.Forward
    accurate: bool
    consumers: int
    interpret: bool
    fused: bool
    kinds: tuple[Kind, ...]
    approx_math: bool
    """Whether its launches take the hardware's approximate transcendentals (`approximate`): never in the
    accurate form."""

    @property
    def how(self) -> algebra.Recompute:
        """How the backward recomputes the forward's weights."""
        return algebra.recompute(self.spec)

    @property
    def scale(self) -> float:
        """The logits' scale."""
        return self.spec.scale

    @property
    def grad_scale(self) -> float:
        """The scale the gradients' cotangents take: the call's whole scale, the logits' times the power its
        query took before its products (`algebra.Forward.prescale`), against the call's own keys for dQ and
        its own queries for dK, so their products stay within the call's own range."""
        return self.spec.scale * self.spec.prescale

    @property
    def scaled(self) -> bool:
        """Whether the call's queries take a power of two for their logits (`by_power`): the fused launch's
        in registers, the keys-first launch's a stream or a stage of its own (`_dkv_step`), dK the call's
        own queries'."""
        return self.spec.prescale != 1.0

    @property
    def summed(self) -> tuple[bool, ...]:
        """Which score parameters the dQ program sums per query (`DQ_GRADS`); the others' gradients the dK/dV
        program writes."""
        return tuple(kind in DQ_GRADS for kind in self.kinds)


def _score_params(family: mosaic.Family, refs, kinds: tuple[Kind, ...], row, head, first, like,
                  k_first=None, *, keys_first: bool = False) -> tuple:
    """A tile's score hook parameters (`_param`), one per kind of `kinds`, read from their refs."""
    return tuple(_param(family, ref, kind, row, head, first, like, k_first, keys_first=keys_first)
                 for ref, kind in zip(refs, kinds, strict=True))


def _scores(walk: _Backward, coords: algebra.Coords, params: tuple, s, dp, peak, total, anchor, delta,
            chosen, *, masked: bool):
    """A backward tile's weights unnormalized and over `total`, its score cotangents (`algebra.grad_scores`)
    and the score hook's slopes (`algebra.tile_logits`), from its products `s` and `dp`, its coordinates, its
    hook parameters `params`, its mask (`algebra.seen` where `masked`) and its queries' statistics."""
    seen = algebra.seen(walk.spec, coords, s) if masked else None
    how = walk.how
    logits, slopes = algebra.tile_logits(coords.hooked(walk.spec.score_mod), s, seen, coords.q_pos,
                                         coords.k_pos, params,
                                         scale=walk.spec.scale if how.factor is None else 1.0, slopes=True)
    # Each query's statistics along a keys-first tile's query axis (`algebra.Coords.queries`); a queries-first
    # tile's the algebra broadcasts at each use, so no broadcast tile stays live across the step (a fused
    # launch at 128 x 64 and width 64 spilled 60 bytes holding them).
    def per_query(x):
        return coords.queries(x, s) if coords.keys_first else x

    inverse_total = per_query(algebra.inverse(total))
    exponentials, weights = algebra.normalized(logits, per_query(peak), inverse_total, how)
    at_anchor = coords.k_pos == coords.queries(chosen, coords.k_pos)
    multiplier = algebra.drawn(coords, coords.q_pos, coords.k_pos)
    cotangents = algebra.grad_scores(exponentials, dp, per_query(anchor), per_query(delta), seen, at_anchor,
                                     inverse_total, multiplier=multiplier)
    return exponentials, weights, cotangents, slopes, seen


def _dq_step(walk: _Backward, operands: tuple, stats: tuple, coords_of, params_of, whole, share, index, slots,
             rows, accumulators):
    """A dQ consumer's key tile `index`: its scores and score cotangents from the key and value slots
    (`slots`) and its queries and output cotangents (`operands`) under its rows' statistics (`stats`), the
    tile's coordinates (`coords_of(k_first)`), its hook parameters (`params_of(like, k_first=k_first)`) and
    whether its block sees it whole (`whole(index)`). The row sums are carried (`rows`); dQ's and the keyed
    sums' products are issued into `accumulators` in place (`algebra.dq_issued`)."""
    q, do = operands
    peak, prior, anchor, delta, chosen = stats
    family, tiles, empty = walk.program.family, walk.program.tiles, jnp.zeros((), jnp.float32)
    shape = (tiles.block_q, tiles.block_k)
    k_slot = slots["keys"]
    k_first = index * tiles.block_k
    s = family.product(q, k_slot.transpose((1, 0)), shape)
    dp = family.product(do, slots["values"].transpose((1, 0)), shape)
    slots.release("values")
    params = params_of(s, k_first=k_first)
    residual, mass, grads, weighed, keyed_mass, keyed_shift = rows
    # dQ's and the keyed sums' places stay empty: their products go to the accumulators.
    sums = algebra.DqSums(empty, residual, mass, grads, weighed, empty, keyed_mass, keyed_shift)

    def issued(*, masked: bool):
        # A tile every query sees whole (`masked` False) takes no mask; the row sums and the parts are taken
        # where the mask is read (`algebra.pulled`).
        scored = _scores(walk, coords_of(k_first), params, s, dp, peak, prior, anchor, delta, chosen,
                         masked=masked)
        exponentials, _, cotangents, slopes, seen = scored
        terms, parts, keyed = algebra.dq_issued(sums, exponentials, cotangents, slopes, seen, k_slot.dtype,
                                                prior, scale=walk.grad_scale, kinds=walk.kinds,
                                                summed=walk.summed, accurate=walk.accurate)
        row_sums = (terms.residual, terms.mass, terms.grads, terms.weighed, terms.keyed_mass,
                    terms.keyed_shift)
        return row_sums, parts, keyed

    # One walk, its tiles' masks chosen a tile at a time: a walk of two loops into one warpgroup accumulator,
    # the first empty, faults on an H100 under jax 0.11.2's Mosaic GPU lowering.
    rows, scaled, keyed = lax.cond(whole(index), functools.partial(issued, masked=False),
                                   functools.partial(issued, masked=True))
    dq_acc, *keyed_accs = accumulators
    # dQ's parts formed past the choice, as each product is issued (`algebra.formed_parts`): the choice's
    # branches yield the scaled cotangents alone, not every part.
    parts = algebra.formed_parts(scaled, k_slot.dtype, accurate=walk.accurate)
    # dQ's (and the keyed sums') products against the program's share of the keys' columns (`share`).
    k_columns = k_slot.at[:, pl.ds(*share)]
    dq_acc = family.register_products(dq_acc, parts, k_columns, walk.interpret)
    if keyed is None:
        return rows, (dq_acc,)
    factor, keyed_parts = keyed
    if factor is None:
        keyed_acc = family.register_products(keyed_accs[0], keyed_parts, k_columns, walk.interpret)
    else:
        # The keyed sums rescaled by their row factor, then the tile's products added in place.
        def rescale(value):
            return value * algebra.expand(factor, value)

        keyed_acc = family.rescaled_products(keyed_accs[0], rescale, keyed_parts, k_columns, walk.interpret)
    return rows, (dq_acc, keyed_acc)


def _dq_launch(query, key, value, dout, peak, prior, anchor, delta, key_index, spans: _Spans | None,
               score_params: tuple, drop: Dropout | None, *, walk: _Backward):
    """`_dq_columns`' launch of each of the config's `columns`, its share of dQ's (and the keyed sums')
    columns, the rows the first's, the sums joined along their widths."""
    columns = walk.program.tiles.columns
    width = query.shape[-1] // columns
    launched = [_dq_columns(query, key, value, dout, peak, prior, anchor, delta, key_index, spans,
                            score_params, drop, walk=walk, share=(column * width, width))
                for column in range(columns)]
    if columns == 1:
        return launched[0]
    dq = jnp.concatenate([out[0] for out in launched], axis=-1)
    keyed_parts = [out[1] for out in launched if out[1] is not None]
    keyed = jnp.concatenate(keyed_parts, axis=-1) if keyed_parts else None
    return dq, keyed, launched[0][2]


def _dq_columns(query, key, value, dout, peak, prior, anchor, delta, key_index, spans: _Spans | None,
                score_params: tuple, drop: Dropout | None, *, walk: _Backward, share: tuple[int, int]):
    """The dQ program's sums over its key tiles (`algebra.dq_issued`), unfinished: dQ's and, in the accurate
    form, the keys' product with the weights (`DqSums.keyed`), `[B, Qp, H, D]` fp32 each; and the rows
    `[4 + 2n, B, H, Qp]` fp32 (n the dQ parameters, `walk.summed`): the cotangents' and weights' row sums, the
    keyed sums' mass and exponent, then each dQ parameter's gradient and its weights' pullback, which
    `dq_finish` takes past the launch. Each query block of a pair is a consumer's, the key and value tiles
    streamed through a ring each. Its sums are its `share` of their columns (`(first, count)`). Returns dQ,
    the keyed sums (None in the fast form) and the rows."""
    batch, queries, heads, width = query.shape
    kv_heads, value_width = key.shape[2], value.shape[-1]
    program, consumers, kinds = walk.program, walk.consumers, walk.kinds
    family = program.family
    block_q, block_k = program.tiles.block_q, program.tiles.block_k
    pair = consumers * block_q
    summed = walk.summed
    rows_count = 4 + 2 * sum(summed)
    leaves, treedef = jax.tree.flatten((spans, score_params, drop))

    def body(*refs, streams):
        (q_ref, k_ref, v_ref, do_ref, peak_ref, prior_ref, anchor_ref, delta_ref, index_ref), refs = (
            refs[:9], refs[9:])
        (spans_of, score_refs, drop_of), refs = treedef.unflatten(refs[:len(leaves)]), refs[len(leaves):]
        dq_ref, refs = refs[0], refs[1:]
        keyed_ref, refs = (refs[0], refs[1:]) if walk.accurate else (None, refs)
        rows_ref, (q_smem, do_smem, out_smem, row_smem, q_loaded, do_loaded) = refs[0], refs[1:]
        row, head = lax.axis_index("batch"), lax.axis_index("heads")
        views = _views_of({"keys": k_ref, "values": v_ref}, row, lax.div(head, heads // kv_heads))
        streams = family.opened(streams, views)
        pair_first = program.query_tile(queries // pair) * pair
        streamed = _streamed([program.reach(pair_first + index * block_q, spans_of, row)[:2]
                              for index in range(consumers)])

        def consumer(index):
            first = pair_first + index * block_q
            rows = (row, pl.ds(first, block_q), head)
            family.loaded(((q_ref.at[rows], q_smem.at[index], q_loaded.at[index]),
                           (do_ref.at[rows], do_smem.at[index], do_loaded.at[index])))
            stats = tuple(_rows(family, ref, row, head, first, block_q)
                          for ref in (peak_ref, prior_ref, anchor_ref, delta_ref, index_ref))
            lo, hi, whole = program.reach(first, spans_of, row)
            nothing = family.zeros((block_q,))
            operands = family.held(q_smem.at[index]), family.held(do_smem.at[index])
            shape = (block_q, block_k)
            # The tile's places formed a step, not held across the walk beside its sums (`_coords`).
            coords_of = functools.partial(_coords, family, spans_of, row, first, shape=shape,
                                          draw=None if drop_of is None else drop_of.draw(row, head))
            params_of = functools.partial(_score_params, family, score_refs, kinds, row, head, first)
            step = functools.partial(_dq_step, walk, operands, stats, coords_of, params_of, whole, share)
            start = (nothing, nothing, (nothing,) * len(kinds), (nothing,) * len(kinds), nothing,
                     nothing + algebra.NO_SHIFT)
            # The tiles before this block's first its partner reads.
            family.passed(streams, streamed[0], lo, first=streamed[0])
            (residual, mass, grads, weighed, keyed_mass, keyed_shift), accumulated = family.walk(
                streams, ((step, lo, hi),), start, ((block_q, share[1]),) * (2 if walk.accurate else 1),
                walk.interpret, first=streamed[0])
            # The tiles past this block's last (a causal block whose diagonal ends first) its partner reads.
            family.passed(streams, hi, streamed[1], first=streamed[0])
            dq, *keyed = accumulated
            written = [(dq, dq_ref.at[rows])]
            if keyed_ref is not None:
                written.append((keyed[0], keyed_ref.at[rows]))
            family.stored(*zip(*written, strict=True), smem=out_smem.at[index])
            # Each row in a row's layout whatever its own (a call without a hook's keyed mass is a constant).
            summed_grads = tuple(part for part, on in zip(grads, summed, strict=True) if on)
            summed_weighed = tuple(part for part, on in zip(weighed, summed, strict=True) if on)
            parts = (residual, mass, keyed_mass, keyed_shift, *summed_grads, *summed_weighed)
            family.stored(tuple(family.as_row(part) for part in parts),
                          tuple(rows_ref.at[place, row, head, pl.ds(first, block_q)]
                                for place in range(rows_count)),
                          smem=row_smem.at[index])

        def producer():
            family.stream(streams, views, *streamed)

        family.roles(consumers, consumer, producer)

    scoped, rings = _dq_allocations(family, program.tiles, width, value_width, query.dtype, program.stages)

    def entry(*refs):
        family.scoped(functools.partial(body, *refs), *scoped, streams=rings)

    sums = jax.ShapeDtypeStruct((batch, queries, heads, share[1]), jnp.float32)
    stats_shape = jax.ShapeDtypeStruct((rows_count, batch, heads, queries), jnp.float32)
    out = family.kernel(
        entry, out_type=(sums, sums, stats_shape) if walk.accurate else (sums, stats_shape),
        grid=(heads, queries // pair, batch), grid_names=("heads", "q_tiles", "batch"), consumers=consumers,
        interpret=walk.interpret, approx_math=walk.approx_math,
        name="attention_dq" + ("" if share[1] == width else f"_columns{share[0] // share[1]}"),
        reducing_warpgroups=reducing(family, consumers, walk.spec.score_mod),
        # Its share of dQ's product, the model's this launch owns (`stated`): its `S` and `dP` recomputed.
        cost=stated(walk.spec, batch, heads, walk.spec.queries or queries, share[1], query, key, value,
                    dout))(
            query, key, value, dout, peak, prior, anchor, delta, key_index, *leaves)
    dq, keyed, rows = out if walk.accurate else (out[0], None, out[1])
    return dq, keyed, rows


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
    logits_slot = None if logits_stage is not None else slots["logits"]
    if count != tiles.block_q:
        q_slot, do_slot = q_slot.at[pl.ds(first_row, count)], do_slot.at[pl.ds(first_row, count)]
        logits_slot = None if logits_slot is None else logits_slot.at[pl.ds(first_row, count)]
    q_first = index * tiles.block_q + first_row
    # The logits from the queries taken by the call's power, the forward's `(Q 2^p) Kᵀ` bit for bit (the
    # power taken after the product passes fp32's largest where the forward's does not, Q 2^120 over keys of
    # 8 at width 64 and scale 1/8; on the keys a key's entry can leave bf16's range, 2^-126 at the scale
    # 2^-10): a stream of their own, or where one passes the block staged a step from the call's own
    # (`BackwardTiles.staged_logits`; a third of the launch at Qwen3-8B, ET-SWAPABL-h100). dK from the call's
    # own, which the power would flush where a query is 2^-126 at the scale 2^-10.
    if logits_stage is not None:
        # The share's rows (32 of a slot's 64 where its consumers split them) read in the layout their
        # copy infers, not the accumulator's (warpgroup MMA's takes whole 64-row tiles).
        logits_slot = family.staged(by_power(q_slot[...], walk.spec.prescale), logits_stage)
    s = family.product(k_smem, logits_slot.transpose((1, 0)), shape)
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
        staged_part = family.staged(part_tile, staged_part)
        dq_stage(q_first, family.product(staged_part.transpose((1, 0)), k_smem, (count, k_smem.shape[-1])))
    if _grading(shares):
        store_tiles(q_first, [piece for kind, piece in zip(kinds, pieces, strict=True) if kind == Kind.TILE])
        key_pieces = [piece for kind, piece in zip(kinds, pieces, strict=True) if kind == Kind.KEY]
        carry = tuple(total + piece for total, piece in zip(carry, key_pieces, strict=True))
    return carry, (dk_acc, dv_acc)


def _grading(shares: tuple) -> bool:
    """Whether a dK/dV program at its head widths' `shares` (`BackwardTiles.columns`) writes the score
    parameters' gradients: the first share's alone, each share's logits and their cotangents the tile's
    whole."""
    return shares[0][0] == 0


KEY_ROWS = MappingProxyType({Kind.KEY: lambda entries: entries.sum(axis=-1)})
"""`algebra.pulled`'s sums of a dK/dV tile, its keys its rows (`_dkv_step`): a key's over its row; the dK/dV
program writes no head's or query's gradient (`DKV_GRADS`)."""


def _unread_tiles(family: mosaic.Family, store_tiles, count: int, shape: tuple[int, int], read: tuple,
                  q_steps: int, block_q: int, first: int = 0) -> None:
    """A dK/dV consumer's `count` tile gradients (`shape` `[bk, its queries]`, those of each `block_q` query
    tile from `first`) over the query tiles its walk does not read (outside `read`, `[lo, hi)` of `q_steps`):
    no query of them sees its keys, so they are zeros (`store_tiles(first query, pieces)`)."""
    zeros = family.zeros(shape)
    lo, hi = read

    @pl.loop(0, lo)
    def _(tile):
        store_tiles(tile * block_q + first, [zeros] * count)

    @pl.loop(hi, q_steps)
    def _(tile):
        store_tiles(tile * block_q + first, [zeros] * count)


def _fused_step(walk: _Backward, resident: tuple, rows_of, staged, coords_of, whole, add_dq, index, slots,
                carry, accumulators):
    """A fused dK/dV consumer's query tile `index`, queries first (`[bq, bk]`, the hookless fast form's
    one launch): its scores and score cotangents from the query and output cotangent slots (`slots`) and its
    resident keys and values (`resident`), under the tile's queries' statistics (`rows_of(first query)`) and
    coordinates (`coords_of(first query)`), a tile every query sees whole (`whole(index)`) unmasked. Its
    weights and score cotangents staged in `staged`, whose transposes are dV's and dK's left operands
    (`algebra.value_operands`, `cotangent_product`'s `aᵀ @ b`) issued into `accumulators` in place, and the
    cotangents times the resident keys the tile's dQ, added into dQ (`add_dq(first query, dQ)`)."""
    k_smem, v_smem = resident
    family, tiles = walk.program.family, walk.program.tiles
    shape = (tiles.block_q, tiles.block_k)
    dk_acc, dv_acc = accumulators
    q_slot, do_slot = slots["queries"], slots["douts"]
    q_first = index * tiles.block_q
    # The tile's queries' statistics read before its products, which their loads run under (ET-A05AB5/6-h100:
    # read past them, 2.32 ms the launch at Qwen3-8B causal; before them, 2.05 against base's 2.09).
    rows = rows_of(q_first)
    # The logits from the query taken by its power in registers (`_Backward.scaled`), dK from the call's own.
    keys = k_smem.transpose((1, 0))
    s = (family.registers_product(by_power(family.tile_of(q_slot), walk.spec.prescale), keys, walk.interpret)
         if walk.scaled else family.product(q_slot, keys, shape))
    dp = family.product(do_slot, v_smem.transpose((1, 0)), shape)

    def scores(*, masked: bool):
        # A tile every query sees whole (`masked` False) takes no mask (`_dq_step`'s choice).
        exponentials, weights, cotangents, _, _ = _scores(walk, coords_of(q_first), (), s, dp, *rows,
                                                          masked=masked)
        return exponentials, weights, cotangents

    exponentials, weights, cotangents = lax.cond(whole(index), functools.partial(scores, masked=False),
                                                 functools.partial(scores, masked=True))
    ((part, _),) = algebra.cotangent_parts(cotangents * walk.grad_scale, q_slot.dtype, accurate=False)
    narrowed = algebra.value_weights(exponentials, weights, do_slot.dtype)
    dv_acc = family.products(dv_acc, family.staged(narrowed, staged.at[0]).transpose((1, 0)), do_slot,
                             walk.interpret)
    cotangents_staged = family.staged(part, staged.at[1])
    dk_acc = family.products(dk_acc, cotangents_staged.transpose((1, 0)), q_slot, walk.interpret)
    # The tile's dQ, its staged cotangents times the resident keys (`algebra.cotangent_product`'s `a @ b`),
    # added into dQ: the query tile's other key blocks add theirs.
    add_dq(q_first, family.product(cotangents_staged, k_smem, (tiles.block_q, k_smem.shape[-1])))
    return carry, (dk_acc, dv_acc)


def _fused_launch(query, key, value, dout, peak, total, anchor, delta, key_index, spans: _Spans | None, *,
                  walk: _Backward) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The fast form's one launch of a hookless call (`BackwardTiles.fused`), on the dQ launch's layout:
    each consumer's `block_k` keys of a query head resident, walked by the query tiles that see them,
    streamed with their output cotangents through a ring each, queries first (`_fused_step`), under the
    first pass's D and the forward's denominator, each tile's dQ added into dQ atomically
    (`Family.added`). Returns dQ `[B, Qp, H, D]`, dK and dV `[B, Kp, H, D]`, fp32."""
    batch, queries, heads, width = query.shape
    keys_padded, kv_heads = key.shape[1], key.shape[2]
    program, consumers = walk.program, walk.consumers
    family = program.family
    block_q, block_k = program.tiles.block_q, program.tiles.block_k
    pair = consumers * block_k
    q_steps = queries // block_q
    leaves, treedef = jax.tree.flatten(spans)

    def body(*refs, streams):
        (q_ref, k_ref, v_ref, do_ref, *stat_refs), refs = refs[:9], refs[9:]
        spans_of, refs = treedef.unflatten(refs[:len(leaves)]), refs[len(leaves):]
        dq_ref, dk_ref, dv_ref, k_smem, v_smem, staged, out_smem, k_loaded, v_loaded = refs
        row, head = lax.axis_index("batch"), lax.axis_index("heads")
        views = _views_of({"queries": q_ref, "douts": do_ref}, row, head)
        streams = family.opened(streams, views)
        pair_first = lax.axis_index("k_tiles") * pair
        streamed = _streamed([program.readers(pair_first + index * block_k, q_steps, spans_of, row)[:2]
                              for index in range(consumers)])

        def rows_of(q_first):
            return tuple(_rows(family, ref, row, head, q_first, block_q) for ref in stat_refs)

        def consumer(index):
            k_first = pair_first + index * block_k
            keys_view = (row, pl.ds(k_first, block_k), lax.div(head, heads // kv_heads))
            family.loaded(((k_ref.at[keys_view], k_smem.at[index], k_loaded.at[index]),
                           (v_ref.at[keys_view], v_smem.at[index], v_loaded.at[index])))
            lo, hi, whole = program.readers(k_first, q_steps, spans_of, row)
            # The tiles before this block's first reader its partner reads.
            family.passed(streams, streamed[0], lo, first=streamed[0])
            coords_of = functools.partial(_coords, family, spans_of, row, k_first=k_first,
                                          shape=(block_q, block_k))
            adds = functools.partial(_dq_added, walk, dq_ref, out_smem, (row, head), index)
            step = functools.partial(_fused_step, walk, (k_smem.at[index], v_smem.at[index]), rows_of,
                                     staged.at[index], coords_of, whole, adds)
            _, (dk, dv) = family.walk(streams, ((step, lo, hi),), (), ((block_k, width), (block_k, width)),
                                      walk.interpret, first=streamed[0])
            # The tiles past this block's last reader its partner reads.
            family.passed(streams, hi, streamed[1], first=streamed[0])
            out = (row, pl.ds(k_first, block_k), head)
            family.stored((dk, dv), (dk_ref.at[out], dv_ref.at[out]),
                          smem=out_smem.at[index, pl.ds(0, block_k)])

        def producer():
            family.stream(streams, views, *streamed)

        family.roles(consumers, consumer, producer)

    scoped, rings = _fused_allocations(family, program.tiles, width, query.dtype,
                                       min(program.tiles.stages, q_steps))

    def entry(*refs):
        family.scoped(functools.partial(body, *refs), *scoped, streams=rings)

    gradient = jax.ShapeDtypeStruct((batch, keys_padded, heads, width), jnp.float32)
    launch = family.kernel(
        entry, out_type=(gradient, gradient), grid=(heads, keys_padded // pair, batch),
        grid_names=("heads", "k_tiles", "batch"), consumers=consumers, interpret=walk.interpret,
        approx_math=walk.approx_math, name="attention_dkv_fused",
        # The model's products this launch owns (`stated`): dQ's, dK's, dV's and dP's; its `S` recomputed.
        cost=stated(walk.spec, batch, heads, walk.spec.queries or queries, 4 * width, query, key, value,
                    dout))
    # dQ added into zeros in place: one copy, or each consumer's own interpreted (`Family.added`).
    copies = consumers if walk.interpret else 1
    dq = jax.new_ref(jnp.zeros((copies, batch, queries, heads, width), jnp.float32))
    dk, dv = launch(query, key, value, dout, peak, total, anchor, delta, key_index, *leaves, dq)
    return jax.freeze(dq).sum(axis=0), dk, dv


def _dq_added(walk: _Backward, dq_ref, out_smem, place: tuple, index, q_first, dq) -> None:
    """A fused dK/dV consumer's tile `dq` added into dQ at its queries from `q_first` (`place` its row and
    head), through its output stage, free until its dK and dV are stored; interpreted, into a copy of the
    consumer's own (`Family.added`)."""
    row, head = place
    rows = dq.shape[0]
    view = dq_ref.at[index if walk.interpret else 0, row, pl.ds(q_first, rows), head]
    walk.program.family.added(dq, view, walk.interpret, smem=out_smem.at[index, pl.ds(0, rows)])


def _resident_loaded(family: mosaic.Family, copies, index, *, shared: bool) -> None:
    """A dK/dV consumer's resident keys and values (`(view, ref, barrier)` each) loaded: `shared` between the
    consumers (`split_queries`), the first loading them and each waiting; else its own."""
    if not shared:
        family.loaded(copies)
        return

    @pl.when(index == 0)
    def _():
        for view, ref, barrier in copies:
            plgpu.copy_gmem_to_smem(view, ref, barrier)

    for _, _, barrier in copies:
        plgpu.barrier_wait(barrier)


def _dkv_stored(family: mosaic.Family, sums: tuple, views: tuple, stages: tuple, index) -> None:
    """A dK/dV consumer's dK and dV (and its key gradients' sums) of its own keys to their `views`, through
    its `stages` (dK's output stage, dV's, the key gradients')."""
    out_smem, dv_smem, rows_smem = stages
    dk, dv, *key_sums = sums
    dk_view, dv_view, *key_views = views
    dk_stage = out_smem.at[index, pl.ds(0, dk.shape[0])]
    if dv_smem is out_smem:
        family.stored((dk, dv), (dk_view, dv_view), smem=dk_stage)
    else:
        family.stored((dk,), (dk_view,), smem=dk_stage)
        family.stored((dv,), (dv_view,), smem=dv_smem.at[index])
    if key_sums:
        # Each key gradient, the walk's row sums, to the `[B, H, Kp]` outputs.
        family.stored(tuple(key_sums), tuple(key_views), smem=rows_smem.at[index])


def _dkv_refs(refs, treedef, leaves: int, graded: list[Kind], *, widths: bool, split: bool, fused: bool
              ) -> tuple:
    """The dK/dV program's refs by role: its nine inputs, its spans' and hook parameters' (`treedef` of
    `leaves`), a fused launch's dQ, its dK and dV outputs, its scoped allocations (`_dkv_scoped`) and its
    key and tile gradients' outputs."""
    inputs, refs = refs[:9], refs[9:]
    unflattened, refs = treedef.unflatten(refs[:leaves]), refs[leaves:]
    dq_ref, refs = (refs[0], refs[1:]) if fused else (None, refs)
    dk_ref, dv_ref, *refs = refs
    grad_refs, refs = refs[:len(graded)], refs[len(graded):]
    keyed = tuple([ref for ref, kind in zip(grad_refs, graded, strict=True) if kind == wanted]
                  for wanted in (Kind.KEY, Kind.TILE))
    return inputs, unflattened, (dq_ref, dk_ref, dv_ref), _dkv_scoped(refs, widths=widths, split=split), keyed


def _shared_views(outputs, graded: list[Kind], block_k: int) -> tuple:
    """dK's, dV's and each key gradient's view (`outputs`, the dK/dV launch's) at the program's key block, the
    one its consumers share (`split_queries`)."""
    dk_ref, dv_ref, *grad_refs = outputs
    row, head = lax.axis_index("batch"), lax.axis_index("heads")
    keys = pl.ds(lax.axis_index("k_tiles") * block_k, block_k)
    key_views = [ref.at[row, head, keys] for ref, kind in zip(grad_refs, graded, strict=True)
                 if kind == Kind.KEY]
    return dk_ref.at[row, keys, head], dv_ref.at[row, keys, head], *key_views


def _summed_stored(family: mosaic.Family, sums: tuple, views: tuple, consumers: int) -> None:
    """The sums of the `consumers` sharing a key block (each over its share of the queries) added in a fixed
    order and stored once to their `views` (`mosaic.summed_across`), each in a scope after the walk's,
    whose shared memory it reuses (`mosaic.summing`)."""
    for value, view in zip(sums, views, strict=True):
        def total(stage, value=value, view=view, **scratch):
            mosaic.summed_across(value, scratch, lax.axis_index("wg"), consumers,
                                 lambda summed: family.stored((summed,), (view,), smem=stage))

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


def _dkv_kernel(walk: _Backward, entry, graded: list[Kind], query, key, value, dout, shares: tuple):
    """The dK/dV program `entry`'s launch (`Family.kernel`) over `query` `[B, Qp, H, D]` and `key` and
    `value` `[B, Kp, Hk, D]`: dK and dV `[B, Kp, H, D]` fp32 out, each its share's columns (`shares`), then
    each of `graded`'s, a key's `[B, H, Kp]` and a tile's keys first `[B, H, Kp, Qp]`, fp32; a program a query
    head's key block a consumer, or one its consumers share (`_holders`)."""
    batch, queries, heads, width = query.shape
    keys_padded, value_width = key.shape[1], value.shape[-1]
    family, consumers = walk.program.family, walk.consumers
    dk, dv = (jax.ShapeDtypeStruct((batch, keys_padded, heads, share), jnp.float32) for _, share in shares)
    grad_shapes = {Kind.KEY: jax.ShapeDtypeStruct((batch, heads, keys_padded), jnp.float32),
                   Kind.TILE: jax.ShapeDtypeStruct((batch, heads, keys_padded, queries), jnp.float32)}
    return family.kernel(
        entry, out_type=(dk, dv, *(grad_shapes[kind] for kind in graded)),
        grid=(heads, keys_padded // (_holders(family, consumers) * walk.program.tiles.block_k), batch),
        grid_names=("heads", "k_tiles", "batch"), consumers=consumers, interpret=walk.interpret,
        approx_math=walk.approx_math,
        name=("attention_dkv_fused" if walk.fused else "attention_dkv")
        + ("" if shares[0][1] == width else f"_columns{shares[0][0] // shares[0][1]}"),
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
    batch, queries, heads, width = query.shape
    kv_heads, value_width = key.shape[2], value.shape[-1]
    program, consumers, kinds = walk.program, walk.consumers, walk.kinds
    family = program.family
    block_q, block_k = program.tiles.block_q, program.tiles.block_k
    # The consumers share one key block, each its share of a slot's queries, where they feed their rings.
    split = split_queries(family)
    pair, count = (block_k, block_q // consumers) if split else (consumers * block_k, block_q)
    q_steps = queries // block_q
    stages = min(program.tiles.stages, q_steps)
    graded = [kind for kind in kinds if kind in DKV_GRADS] if _grading(shares) else []
    # A tile parameter's entries keys first, as the program's tiles hold them (`_param`).
    staged_logits = program.tiles.staged_logits
    leaves, treedef = jax.tree.flatten((spans, _keys_last(score_params, kinds), drop,
                                        None if staged_logits else scaled))

    def body(*refs, streams):
        inputs, unflattened, outputs, held, (key_refs, tile_refs) = _dkv_refs(
            refs, treedef, len(leaves), graded, widths=value_width == width, split=split, fused=walk.fused)
        (spans_of, score_refs, drop_of, l_ref), (dq_ref, dk_ref, dv_ref) = unflattened, outputs
        q_ref, k_ref, v_ref, do_ref, *stat_refs = inputs
        k_smem, v_smem, out_smem, dv_smem, rows_smem, k_loaded, v_loaded, *rest = held
        fused_stages, logits_stages = (rest[:2], rest[2:]) if walk.fused else ((), rest)
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
            _resident_loaded(family, copies, index, shared=split)
            lo, hi, _ = program.readers(k_first, q_steps, spans_of, row)
            part = (index * count, count) if split else (0, block_q)

            def store_tiles(first, pieces):
                # Each tile's gradient entries straight to their place in the keys-first `[B, H, Kp, Qp]`
                # outputs.
                for ref, piece in zip(tile_refs, pieces, strict=True):
                    ref[row, head, pl.ds(k_first, block_k), pl.ds(first, piece.shape[1])] = piece

            if not split:
                # The tiles before this block's first reader its partner reads.
                family.passed(streams, streamed[0], lo, first=streamed[0])
            coords_of = functools.partial(_coords, family, spans_of, row, k_first=k_first, keys_first=True,
                                          draw=None if drop_of is None else drop_of.draw(row, head))
            params_of = functools.partial(_score_params, family, score_refs, kinds, row, head,
                                          k_first=k_first, keys_first=True)
            resident = k_smem.at[holder], v_smem.at[holder]
            stage = logits_stages[0].at[index] if logits_stages else None
            adds = None
            if walk.fused:
                parts_stage, dq_stages = fused_stages
                adds = (parts_stage.at[index],
                        functools.partial(_dq_added, walk, dq_ref, dq_stages, (row, head), index))
            step = functools.partial(_dkv_step, walk, resident, rows_of, coords_of, params_of, store_tiles,
                                     shares, part, stage, adds)
            # Its consumers' own loads of its query stream where they feed it (`Family.fed`).
            fed = (family.fed(streams, views, lo, hi, index)
                   if isinstance(family, mosaic.ConsumerFed) else streams)
            key_sums, (dk, dv) = family.walk(fed, ((step, lo, hi),),
                                             tuple(family.zeros((block_k,)) for _ in key_refs),
                                             tuple((block_k, share) for _, share in shares), walk.interpret,
                                             first=lo if split else streamed[0])
            if not split:
                # The tiles past this block's last reader its partner reads.
                family.passed(streams, hi, streamed[1], first=streamed[0])
            if tile_refs:
                _unread_tiles(family, store_tiles, len(tile_refs), (block_k, count), (lo, hi), q_steps,
                              block_q, part[0])
            if split:
                # The consumers' sums, added in a scope after the walk's (`_summed_stored`).
                return dk, dv, *key_sums
            out = (row, pl.ds(k_first, block_k), head)
            key_views = tuple(ref.at[row, head, pl.ds(k_first, block_k)] for ref in key_refs)
            _dkv_stored(family, (dk, dv, *key_sums), (dk_ref.at[out], dv_ref.at[out], *key_views),
                        (out_smem, dv_smem, rows_smem), index)
            return None

        def producer():
            family.stream(streams, views, *streamed)

        return family.roles(consumers, consumer, producer)

    scoped, rings = _dkv_allocations(family, program.tiles, width, value_width, query.dtype, stages=stages,
                                     columns=program.tiles.columns)

    def entry(*refs):
        summed = family.scoped(functools.partial(body, *refs), *scoped, streams=rings)
        if summed is not None:
            outputs = refs[9 + len(leaves) + walk.fused:]
            _summed_stored(family, summed, _shared_views(outputs, graded, block_k), consumers)

    launch = _dkv_kernel(walk, entry, graded, query, key, value, dout, shares)
    operands = (query, key, value, dout, peak, total, anchor, delta, key_index, *leaves)
    if not walk.fused:
        dk, dv, *grads = launch(*operands)
        return dk, dv, *_keys_last(grads, graded)
    # dQ added into zeros in place: one copy, or each consumer's own interpreted (`Family.added`).
    copies = consumers if walk.interpret else 1
    dq = jax.new_ref(jnp.zeros((copies, batch, queries, heads, width), jnp.float32))
    dk, dv = launch(*operands, dq)
    return jax.freeze(dq).sum(axis=0), dk, dv


def _centre_launch(query, key, value, dout, peak, spans: _Spans | None, score_params: tuple, drop: Dropout, *,
                   walk: _Backward) -> algebra.Centre:
    """Dropout's first pass on this mainloop (`algebra.dropout_centre`): each query tile's walk over its keys
    (a consumer's, as the dQ program's), each tile's logits `exp(s - m)` and dropped products
    (`algebra.dropped_products` of `dO Vᵀ` and the call's draw) taken into the query's `Centre`
    (`algebra.centre_step`). Returns the centre and its seen keys `[B, H, Qp]` and each running sum
    `[k, B, H, Qp]` (`precision.running_sum`), which `algebra.centre_finish` takes past the launch."""
    batch, queries, heads, width = query.shape
    kv_heads = key.shape[2]
    program, consumers, kinds = walk.program, walk.consumers, walk.kinds
    family = program.family
    block_q, block_k = program.tiles.block_q, program.tiles.block_k
    pair = consumers * block_q
    exponent = algebra.centre_scale(walk.spec.keys)
    leaves, treedef = jax.tree.flatten((spans, score_params, drop))
    # The centre, whether a key is seen, and each running sum's parts (`precision.running_sum`).
    parts = running_sum((1,), jnp.float32).shape[0]
    fields = 2 + 2 * parts

    def body(*refs, streams):
        (q_ref, k_ref, v_ref, do_ref, peak_ref), refs = refs[:5], refs[5:]
        (spans_of, score_refs, drop_of), refs = treedef.unflatten(refs[:len(leaves)]), refs[len(leaves):]
        centre_ref, (q_smem, do_smem, row_smem, q_loaded, do_loaded) = refs[0], refs[1:]
        row, head = lax.axis_index("batch"), lax.axis_index("heads")
        views = _views_of({"keys": k_ref, "values": v_ref}, row, lax.div(head, heads // kv_heads))
        streams = family.opened(streams, views)
        pair_first = program.query_tile(queries // pair) * pair
        streamed = _streamed([program.reach(pair_first + index * block_q, spans_of, row)[:2]
                              for index in range(consumers)])

        def consumer(index):
            first = pair_first + index * block_q
            rows = (row, pl.ds(first, block_q), head)
            family.loaded(((q_ref.at[rows], q_smem.at[index], q_loaded.at[index]),
                           (do_ref.at[rows], do_smem.at[index], do_loaded.at[index])))
            peak = _rows(family, peak_ref, row, head, first, block_q)
            lo, hi, _ = program.reach(first, spans_of, row)
            shape = (block_q, block_k)
            draw = drop_of.draw(row, head)
            q, do = family.held(q_smem.at[index]), family.held(do_smem.at[index])

            def step(index, slots, state, accumulators):
                k_first = index * block_k
                s = family.product(q, slots["keys"].transpose((1, 0)), shape)
                dp = family.product(do, slots["values"].transpose((1, 0)), shape)
                slots.release("values")
                coords = _coords(family, spans_of, row, first, k_first, shape, draw=draw)
                params = _score_params(family, score_refs, kinds, row, head, first, s, k_first)
                # Every entry's visibility, a tile whole too (`algebra.seen` gives None for one no mask cuts).
                seen = algebra.seen(walk.spec, coords, s)
                seen = jnp.ones(shape, bool) if seen is None else seen
                logits, _ = algebra.tile_logits(walk.spec.score_mod, s, seen, coords.q_pos, coords.k_pos,
                                                params, scale=walk.spec.scale if walk.how.factor is None
                                                else 1.0, slopes=False)
                weights = algebra.below(logits, algebra.expand(peak, logits), walk.how)
                products = algebra.dropped_products(dp, algebra.drawn(coords, coords.q_pos, coords.k_pos))
                centred = algebra.centre_step(state, weights, products, seen, coords.k_pos, exponent)
                return centred, accumulators

            family.passed(streams, streamed[0], lo, first=streamed[0])
            # Every field a row vector of zeros (`algebra.centre_start`), `seen` false, each running sum
            # its leading axis's parts (`precision.added_to`): the layouts hold no unit axis.
            zero = family.zeros((block_q,))
            parts = running_sum((1,), jnp.float32).shape[0]
            start = algebra.Centre(zero, zero > 0, (zero,) * parts, (zero,) * parts)
            centred, _ = family.walk(streams, ((step, lo, hi),), start, (), walk.interpret, first=streamed[0])
            family.passed(streams, hi, streamed[1], first=streamed[0])
            rows_of = [part.astype(jnp.float32) for part in jax.tree.leaves(centred)]
            family.stored(tuple(family.as_row(part) for part in rows_of),
                          tuple(centre_ref.at[place, row, head, pl.ds(first, block_q)]
                                for place in range(fields)), smem=row_smem.at[index])

        def producer():
            family.stream(streams, views, *streamed)

        family.roles(consumers, consumer, producer)

    scoped = [plgpu.SMEM((consumers, block_q, width), query.dtype),
              plgpu.SMEM((consumers, block_q, dout.shape[-1]), dout.dtype),
              plgpu.SMEM((consumers, block_q), jnp.float32),
              plgpu.Barrier(num_barriers=consumers), plgpu.Barrier(num_barriers=consumers)]
    rings = family.streams({"keys": ((block_k, width), key.dtype), "values": ((block_k, value.shape[-1]),
                                                                              value.dtype)},
                           program.stages, consumers)

    def entry(*refs):
        family.scoped(functools.partial(body, *refs), *scoped, streams=rings)

    out = family.kernel(
        entry, out_type=jax.ShapeDtypeStruct((fields, batch, heads, queries), jnp.float32),
        grid=(heads, queries // pair, batch), grid_names=("heads", "q_tiles", "batch"), consumers=consumers,
        interpret=walk.interpret, approx_math=False, name="attention_centre",
        cost=stated(walk.spec, batch, heads, walk.spec.queries or queries, 0, query, key, value, dout)
    )(query, key, value, dout, peak, *leaves)
    return algebra.Centre(out[0], out[1] > 0, out[2:2 + parts], out[2 + parts:])


def _launch_backward(query, key, value, dout, peak0, total0, initial_term, maximum, key_index, anchor_value,
                     shift, others, own, others_total, forward_total, start, end, query_ids, key_ids,
                     score_params, *, score_mod, score_kinds, score_dtypes, scale: float, prescale: float,
                     spec: algebra.Forward, accurate: bool, tiles: BackwardTiles, family: mosaic.Family,
                     interpret: bool, drop: Dropout | None = None):
    """`gradients`' launch on this mainloop: D from the forward's split sums, each query's anchor key as the
    first pass chose it. Two launches: the dQ program's sums, their finish (`algebra.dq_finish`) and the dK/dV
    program under its refined D, whose cotangents sum to 0 where the first pass's do not (a small weight on a
    large value, bf16: dK's worst row 2.5e1 at the first pass's D, where check 1 allows 1.55e-2), the accurate
    form's always, and a score hook's and an initial state's, whose gradients need the refined D
    (`algebra.dq_step`). The fast form's one launch (`tiles.fused`, a hookless call's): the dK/dV program
    adding each tile's dQ atomically, under the first pass's D and the forward's denominator, the derived
    default of a hookless fast-form call outside deterministic mode (`MosaicBackward.derived`: 1.05 to 1.4
    times as fast as two launches, FUSEDFORM-h100; 384 MB of temporaries against 258 at llama-d128 2 x 4096,
    its dQ fp32); two launches, in a fixed order, otherwise. Returns what `gradients` takes: dQ, the softmax's
    rows, each query's anchor key, the dQ parameters' gradients (`DQ_GRADS`; a head's as per-query sums
    `[1, B, H, 1]`, which `gradients` sums over the batch), then dK, dV and the dK/dV parameters'
    (`DKV_GRADS`). The query arrives the call's own: its logits are those of the query taken by `prescale`
    (`by_power`) at `scale`, the scale left (`gradients`), which `spec` takes."""
    spec = dataclasses.replace(spec, scale=scale, prescale=prescale)
    width, value_width = query.shape[-1], value.shape[-1]
    _, state_total = algebra.off_anchor_start(peak0, total0, maximum, value_width, shift)
    cotangent = jnp.swapaxes(dout, 1, 2).astype(jnp.float32)
    anchor, delta, chosen = algebra.split_delta(others, own, others_total, forward_total, state_total,
                                                jnp.swapaxes(anchor_value, 1, 2), cotangent, key_index, shift)
    # The family's swizzled tiles hold whole rows (`Family.column_unit`).
    query, key, value, dout = (_columns(array, family.column_unit(array.dtype))
                               for array in (query, key, value, dout))
    steps = key.shape[1] // tiles.block_k
    program = _Program(family, tiles, steps, min(tiles.stages, steps), spec.keys, spec.form == "causal",
                       spec.queries)
    hooks = Hooks(score_mod, tuple(map(Param, score_params, score_kinds)))
    # The accurate form's exponentials exact: the hardware's ex2.approx flushes a subnormal weight to zero,
    # whose product with a large score cotangent the reference keeps (row 2022, ET-SUBFIX2-h100).
    approx_math = not accurate and approximate(hooks, (tiles.block_q, tiles.block_k), value_width,
                                               differentiated=True)
    walk = _Backward(program, spec, accurate, tiles.consumers, interpret, tiles.fused, score_kinds,
                     approx_math)
    padded = PaddedCall(query, key, value, start, end, query_ids, key_ids, (), (), query.shape[1], spec.keys)
    spans = _spans_of(spec, padded, tiles.block_q, tiles.block_k)
    # The queries taken by the call's power: the logits' operand of the two launches (`_dkv_step`).
    scaled = by_power(query, prescale)
    if drop is not None:
        # A dropping call's D its dropped products' own, centred about each query's least one
        # (`algebra.dropout_centre`): no anchor, and no key chosen.
        centre = _centre_launch(scaled, key, value, dout, maximum, spans, score_params,
                                drop, walk=walk)
        anchor = jnp.zeros_like(anchor)
        delta = algebra.centre_finish(centre, state_total, initial_term, algebra.centre_scale(spec.keys))
        chosen = jnp.full_like(chosen, -1)
    # The dK/dV launch on its own layout (`BackwardTiles.dkv`).
    dkv_tiles = tiles.dkv
    dkv_walk = dataclasses.replace(walk, program=dataclasses.replace(
        program, family=bundles.named(dkv_tiles.family, query.dtype), tiles=dkv_tiles),
        consumers=dkv_tiles.consumers)
    if walk.fused and dkv_tiles != tiles:
        # The fused launch keys first on its own layout (`BackwardTiles.dkv`): its consumers' shares of a
        # slot's queries each a warpgroup MMA's rows, dQ added a tile at a time.
        dq, dk, dv = _dkv_launch(query, key, value, dout, maximum, forward_total, anchor, delta, chosen,
                                 spans, score_params, drop, scaled, walk=dkv_walk)
    elif walk.fused:
        dq, dk, dv = _fused_launch(query, key, value, dout, maximum, forward_total, anchor, delta, chosen,
                                   spans, walk=walk)
    if walk.fused:
        mass = algebra.split_mass(own, others_total, forward_total, shift)
        return (dq[..., :width], delta, anchor, maximum, forward_total, mass, chosen, dk[..., :width],
                dv[..., :value_width])
    # The dQ program reads its queries for their logits alone.
    dq_sum, keyed, rows = _dq_launch(scaled, key, value, dout, maximum, forward_total,
                                     anchor, delta, chosen, spans, score_params, drop, walk=walk)
    count = sum(walk.summed)
    dq_swapped = jnp.swapaxes(dq_sum, 1, 2)
    # The fast form's keyed sums are never read (`dq_finish`): a placeholder there.
    keyed = jnp.zeros_like(dq_swapped) if keyed is None else jnp.swapaxes(keyed, 1, 2)
    sums = algebra.DqSums(dq_swapped, rows[0], rows[1], tuple(rows[4:4 + count]), tuple(rows[4 + count:]),
                          keyed, rows[2], rows[3])
    dq, refined, denominator, mass, dq_grads = algebra.dq_finish(sums, forward_total, state_total, peak0,
                                                                  (total0, initial_term), maximum, anchor,
                                                                  delta, accurate=accurate)
    dk, dv, *key_tile = _dkv_launch(query, key, value, dout, maximum, denominator, anchor, refined, chosen,
                                    spans, score_params, drop, scaled, walk=dkv_walk)
    summed_kinds = [kind for kind in score_kinds if kind in DQ_GRADS]
    head_query = tuple(grad.sum(axis=-1)[None, :, :, None] if kind == Kind.HEAD else grad
                       for grad, kind in zip(dq_grads, summed_kinds, strict=True))
    return (jnp.swapaxes(dq, 1, 2)[..., :width], refined, anchor, maximum, denominator, mass, chosen,
            *head_query, dk[..., :width], dv[..., :value_width], *key_tile)
