"""The attention template's Mosaic GPU backward (`attention_mosaic.MosaicBackward`'s launches): D about each
query's anchor from a training forward's split sums (`algebra.split_delta`), then a dQ launch and a dK/dV
launch, `dq_finish`'s refinement between them, in either form, or the fast form's one launch, the dK/dV
program adding each tile's dQ (`_launch_backward`). Each launch's programs walk the bundle's streams as the
forward's do (`attention_mosaic_program`), the algebra's steps between its products."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import attention_algebra as algebra, mosaic
from dew.nn.kernels.templates.attention import DKV_GRADS, DQ_GRADS, Hooks, Kind, PaddedCall, Param, by_power
from dew.nn.kernels.templates.attention_mosaic_program import (
    BackwardTiles,
    _columns,
    _coords,
    _param,
    _parts,
    _Program,
    _rows,
    _Spans,
    _spans_of,
    _streamed,
    _views_of,
    approximate,
    stated,
)


def _dq_allocations(family: mosaic.Family, config: BackwardTiles, width: int, value_width: int, dtype,
                    stages: int) -> tuple[list, dict]:
    """The dQ program's allocations as `Family.scoped` takes them, in `_dq_launch`'s order: its consumers'
    query and output-cotangent stages, their dQ output stage (fp32) and their rows' stage, the two loads'
    barriers, and its key and value streams in `stages` slots (`Family.streams`)."""
    consumers, block_q, block_k = config.consumers, config.block_q, config.block_k
    scoped = [plgpu.SMEM((consumers, block_q, width), dtype),
              plgpu.SMEM((consumers, block_q, value_width), dtype),
              plgpu.SMEM((consumers, block_q, width), jnp.float32),
              plgpu.SMEM((consumers, block_q), jnp.float32),
              plgpu.Barrier(num_barriers=consumers), plgpu.Barrier(num_barriers=consumers)]
    blocks = {"keys": ((block_k, width), dtype), "values": ((block_k, value_width), dtype)}
    return scoped, family.streams(blocks, stages, consumers)


def _dkv_allocations(family: mosaic.Family, config: BackwardTiles, width: int, value_width: int, dtype, *,
                     parts: int, stages: int) -> tuple[list, dict]:
    """The dK/dV program's allocations as `Family.scoped` takes them, in `_dkv_launch`'s order: its
    consumers' resident keys and values; a tile's weights and score cotangents' `parts` staged for the
    transposed products (`algebra.value_operands`, `algebra.cotangent_parts`); dK's and dV's output stage,
    which a fused launch's dQ tiles take too (fp32); each consumer's scratch for the key gradients' sums
    across its warps (`Family.column_sums`); the two loads' barriers; and its query and output-cotangent
    streams in `stages` slots (`Family.streams`)."""
    consumers, block_q, block_k = config.consumers, config.block_q, config.block_k
    scoped = [plgpu.SMEM((consumers, block_k, width), dtype),
              plgpu.SMEM((consumers, block_k, value_width), dtype),
              plgpu.SMEM((consumers, 1 + parts, block_q, block_k), dtype),
              plgpu.SMEM((consumers, max(block_q, block_k), width), jnp.float32),
              plgpu.SMEM((consumers, family.column_scratch), jnp.float32),
              plgpu.Barrier(num_barriers=consumers), plgpu.Barrier(num_barriers=consumers)]
    blocks = {"queries": ((block_q, width), dtype), "douts": ((block_q, value_width), dtype)}
    return scoped, family.streams(blocks, stages, consumers)


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
        """Whether the dK/dV program's logits take the query by its power (`by_power`), formed in registers
        from the call's own streamed one, which dK takes."""
        return self.spec.prescale != 1.0

    @property
    def summed(self) -> tuple[bool, ...]:
        """Which score parameters the dQ program sums per query (`DQ_GRADS`); the others' gradients the dK/dV
        program writes."""
        return tuple(kind in DQ_GRADS for kind in self.kinds)


def _score_params(family: mosaic.Family, refs, kinds: tuple[Kind, ...], row, head, first, like,
                  k_first=None) -> tuple:
    """A tile's score hook parameters (`_param`), one per kind of `kinds`, read from their refs."""
    return tuple(_param(family, ref, kind, row, head, first, like, k_first)
                 for ref, kind in zip(refs, kinds, strict=True))


def _scores(walk: _Backward, coords: algebra.Coords, masked: bool, params: tuple, s, dp, peak, total, anchor,
            delta, chosen):
    """A backward tile's weights unnormalized and over `total`, its score cotangents (`algebra.grad_scores`)
    and the score hook's slopes (`algebra.tile_logits`), from its products `s` and `dp`, its coordinates, its
    hook parameters `params`, its mask (`algebra.seen` where `masked`) and its queries' statistics."""
    seen = algebra.seen(walk.spec, coords, s) if masked else None
    how = walk.how
    logits, slopes = algebra.tile_logits(walk.spec.score_mod, s, seen, coords.q_pos, coords.k_pos, params,
                                         scale=walk.spec.scale if how.factor is None else 1.0, slopes=True)
    exponentials, weights = algebra.normalized(logits, peak, total, how)
    at_anchor = coords.k_pos == algebra.expand(chosen, coords.k_pos)
    cotangents = algebra.grad_scores(exponentials, dp, anchor, delta, seen, at_anchor, algebra.inverse(total))
    return exponentials, weights, cotangents, slopes, seen


def _dq_step(walk: _Backward, operands: tuple, stats: tuple, coords_of, params_of, whole, index, slots, rows,
             accumulators):
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
        scored = _scores(walk, coords_of(k_first), masked, params, s, dp, peak, prior, anchor, delta, chosen)
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
    dq_acc = family.register_products(dq_acc, parts, k_slot, walk.interpret)
    if keyed is None:
        return rows, (dq_acc,)
    factor, keyed_parts = keyed
    if factor is None:
        keyed_acc = family.register_products(keyed_accs[0], keyed_parts, k_slot, walk.interpret)
    else:
        # The keyed sums rescaled by their row factor, then the tile's products added in place.
        def rescale(value):
            return value * algebra.expand(factor, value)

        keyed_acc = family.rescaled_products(keyed_accs[0], rescale, keyed_parts, k_slot, walk.interpret)
    return rows, (dq_acc, keyed_acc)


def _dq_launch(query, key, value, dout, peak, prior, anchor, delta, key_index, spans: _Spans | None,
               score_params: tuple, *, walk: _Backward):
    """The dQ program's sums over its key tiles (`algebra.dq_issued`), unfinished: dQ's and, in the accurate
    form, the keys' product with the weights (`DqSums.keyed`), `[B, Qp, H, D]` fp32 each; and the rows
    `[4 + 2n, B, H, Qp]` fp32 (n the dQ parameters, `walk.summed`): the cotangents' and weights' row sums, the
    keyed sums' mass and exponent, then each dQ parameter's gradient and its weights' pullback, which
    `dq_finish` takes past the launch. Each query block of a pair is a consumer's, the key and value tiles
    streamed through a ring each. Returns dQ, the keyed sums (None in the fast form) and the rows."""
    batch, queries, heads, width = query.shape
    kv_heads, value_width = key.shape[2], value.shape[-1]
    program, consumers, kinds = walk.program, walk.consumers, walk.kinds
    family = program.family
    block_q, block_k = program.tiles.block_q, program.tiles.block_k
    pair = consumers * block_q
    summed = walk.summed
    rows_count = 4 + 2 * sum(summed)
    leaves, treedef = jax.tree.flatten((spans, score_params))

    def body(*refs, streams):
        (q_ref, k_ref, v_ref, do_ref, peak_ref, prior_ref, anchor_ref, delta_ref, index_ref), refs = (
            refs[:9], refs[9:])
        (spans_of, score_refs), refs = treedef.unflatten(refs[:len(leaves)]), refs[len(leaves):]
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
            coords_of = functools.partial(_coords, family, spans_of, row, first, shape=shape)
            params_of = functools.partial(_score_params, family, score_refs, kinds, row, head, first)
            step = functools.partial(_dq_step, walk, operands, stats, coords_of, params_of, whole)
            start = (nothing, nothing, (nothing,) * len(kinds), (nothing,) * len(kinds), nothing,
                     nothing + algebra.NO_SHIFT)
            # The tiles before this block's first its partner reads.
            family.passed(streams, streamed[0], lo, first=streamed[0])
            (residual, mass, grads, weighed, keyed_mass, keyed_shift), accumulated = family.walk(
                streams, ((step, lo, hi),), start, ((block_q, width),) * (2 if walk.accurate else 1),
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

    sums = jax.ShapeDtypeStruct((batch, queries, heads, width), jnp.float32)
    stats_shape = jax.ShapeDtypeStruct((rows_count, batch, heads, queries), jnp.float32)
    out = family.kernel(
        entry, out_type=(sums, sums, stats_shape) if walk.accurate else (sums, stats_shape),
        grid=(heads, queries // pair, batch), grid_names=("heads", "q_tiles", "batch"), consumers=consumers,
        interpret=walk.interpret, approx_math=walk.approx_math, name="attention_dq",
        # dQ's product, the model's this launch owns (`stated`): its `S` and `dP` recomputed.
        cost=stated(walk.spec, batch, heads, walk.spec.queries or queries, width, query, key, value, dout))(
            query, key, value, dout, peak, prior, anchor, delta, key_index, *leaves)
    dq, keyed, rows = out if walk.accurate else (out[0], None, out[1])
    return dq, keyed, rows


def _dkv_step(walk: _Backward, resident: tuple, rows_of, buffers, coords_of, params_of, whole, add_dq,
              store_tiles, sums_scratch, index, slots, carry, accumulators):
    """A dK/dV consumer's query tile `index`: its scores and score cotangents from the query and output
    cotangent slots (`slots`) and its resident keys and values (`resident`) under the tile's rows' statistics
    (`rows_of(first query)`), its coordinates (`coords_of(first query)`), its hook parameters
    (`params_of(first query, like)`) and whether it sees the keys whole (`whole(index)`). dV's and dK's
    products are issued into `accumulators` in place, each part staged in `buffers` as its transposed
    product's left operand; the key gradients' column sums (`carry`) take the pulled cotangents' key
    sums (`Family.column_sums` through the consumer's `sums_scratch`), and the tile gradients go to
    `store_tiles(first query, pieces)`."""
    k_smem, v_smem = resident
    family, tiles, kinds = walk.program.family, walk.program.tiles, walk.kinds
    shape = (tiles.block_q, tiles.block_k)
    dk_acc, dv_acc = accumulators
    q_slot, do_slot = slots["queries"], slots["douts"]
    q_first = index * tiles.block_q
    peak, denominator, centre, rest, chosen = rows_of(q_first)
    # The logits from the query taken by its power in registers (`_Backward.scaled`), dK from the call's own.
    keys = k_smem.transpose((1, 0))
    s = (family.registers_product(by_power(family.tile_of(q_slot), walk.spec.prescale), keys, walk.interpret)
         if walk.scaled else family.product(q_slot, keys, shape))
    dp = family.product(do_slot, v_smem.transpose((1, 0)), shape)
    params = params_of(q_first, s)

    def scores(*, masked: bool):
        # A tile every query sees whole (`masked` False) takes no mask; the pulled cotangents and the hook's
        # pieces are taken where the mask is read (`algebra.pulled`).
        exponentials, weights, cotangents, slopes, seen = _scores(
            walk, coords_of(q_first), masked, params, s, dp, peak, denominator, centre, rest, chosen)
        column_sums = functools.partial(family.column_sums, scratch=sums_scratch, interpret=walk.interpret)
        logits, pieces = algebra.pulled(cotangents, slopes, seen, kinds, column_sums)
        return exponentials, weights, logits, pieces

    # The tiles a query block sees whole unmasked, the rest masked, chosen a tile at a time in one walk
    # (`_dq_step`).
    exponentials, weights, logits, pieces = lax.cond(whole(index), functools.partial(scores, masked=False),
                                                     functools.partial(scores, masked=True))
    narrowed = algebra.value_weights(exponentials, weights, do_slot.dtype)
    parts = algebra.cotangent_parts(logits * walk.grad_scale, q_slot.dtype, accurate=walk.accurate)
    # Each staged part's transpose the left operand (`algebra.cotangent_product`'s `aᵀ @ b`).
    dv_acc = family.staged_products(dv_acc, ((narrowed, None),), do_slot, buffers.at[pl.ds(0, 1)],
                                    walk.interpret, transposed=True)
    dk_acc = family.staged_products(dk_acc, parts, q_slot, buffers.at[pl.ds(1, len(parts))], walk.interpret,
                                    transposed=True)
    if add_dq is not None:
        # The tile's dQ, its staged cotangents times the resident keys (`algebra.cotangent_product`'s
        # `a @ b`), added into dQ: the query tile's other key blocks add theirs.
        add_dq(q_first, family.product(buffers.at[1], k_smem, (tiles.block_q, k_smem.shape[-1])))
    store_tiles(q_first, [piece for kind, piece in zip(kinds, pieces, strict=True) if kind == Kind.TILE])
    key_pieces = [piece for kind, piece in zip(kinds, pieces, strict=True) if kind == Kind.KEY]
    # The staged parts' reads done (the walk's settling) before the next tile's overwrite them.
    return tuple(total + piece for total, piece in zip(carry, key_pieces, strict=True)), (dk_acc, dv_acc)


def _unread_tiles(family: mosaic.Family, store_tiles, count: int, shape: tuple[int, int], read: tuple,
                  q_steps: int) -> None:
    """A dK/dV consumer's `count` tile gradients over the query tiles its walk does not read (outside `read`,
    `[lo, hi)` of `q_steps`): no query of them sees its keys, so they are zeros (`store_tiles(first query,
    pieces)`)."""
    zeros = family.zeros(shape)
    lo, hi = read

    @pl.loop(0, lo)
    def _(tile):
        store_tiles(tile * shape[0], [zeros] * count)

    @pl.loop(hi, q_steps)
    def _(tile):
        store_tiles(tile * shape[0], [zeros] * count)


def _dkv_launch(query, key, value, dout, peak, total, anchor, delta, key_index, spans: _Spans | None,
                score_params: tuple, *, walk: _Backward) -> tuple[jax.Array, ...]:
    """The dK/dV program under the dQ program's softmax: each consumer's `block_k` keys of a query head,
    resident, walked by the query tiles that see them, streamed with their output cotangents through a ring
    each; a tile's weights and score cotangents (the algebra's, rows the queries) staged in shared memory as
    the transposed products' left operands (`algebra.value_operands`, `cotangent_product`). Returns the
    query head's dK and dV `[B, Kp, H, D]` fp32, then each key and tile gradient of the score hook in kind
    order (`DKV_GRADS`: keys `[B, H, Kp]`, tiles `[B, H, Qp, Kp]`); `fused`, dQ `[B, Qp, H, D]` fp32 before
    them, each tile's staged cotangents times the resident keys added into it (`Family.added`). A key or
    tile gradient is written by the one program that holds its keys, straight from the registers; the tiles
    its walk does not read are zeros."""
    batch, queries, heads, width = query.shape
    keys_padded, kv_heads, value_width = key.shape[1], key.shape[2], value.shape[-1]
    program, consumers, kinds = walk.program, walk.consumers, walk.kinds
    family = program.family
    block_q, block_k = program.tiles.block_q, program.tiles.block_k
    pair = consumers * block_k
    q_steps = queries // block_q
    stages = min(program.tiles.stages, q_steps)
    graded = [kind for kind in kinds if kind in DKV_GRADS]
    leaves, treedef = jax.tree.flatten((spans, score_params))

    def body(*refs, streams):
        (q_ref, k_ref, v_ref, do_ref, peak_ref, total_ref, anchor_ref, delta_ref, index_ref), refs = (
            refs[:9], refs[9:])
        (spans_of, score_refs), refs = treedef.unflatten(refs[:len(leaves)]), refs[len(leaves):]
        dq_ref, refs = (refs[0], refs[1:]) if walk.fused else (None, refs)
        dk_ref, dv_ref, *refs = refs
        grad_refs, refs = refs[:len(graded)], refs[len(graded):]
        k_smem, v_smem, staged, out_smem, sums_smem, k_loaded, v_loaded = refs
        key_refs = [ref for ref, kind in zip(grad_refs, graded, strict=True) if kind == Kind.KEY]
        tile_refs = [ref for ref, kind in zip(grad_refs, graded, strict=True) if kind == Kind.TILE]
        row, head = lax.axis_index("batch"), lax.axis_index("heads")

        def add_dq(dq_ref, index, q_first, dq):
            # A tile's dQ through the consumer's output stage, free until its dK and dV are stored;
            # interpreted, into a copy of the consumer's own (`Family.added`).
            view = dq_ref.at[index if walk.interpret else 0, row, pl.ds(q_first, block_q), head]
            family.added(dq, view, walk.interpret, smem=out_smem.at[index, pl.ds(0, block_q)])

        views = _views_of({"queries": q_ref, "douts": do_ref}, row, head)
        streams = family.opened(streams, views)
        pair_first = lax.axis_index("k_tiles") * pair
        streamed = _streamed([program.readers(pair_first + index * block_k, q_steps, spans_of, row)[:2]
                              for index in range(consumers)])

        def rows_of(q_first):
            return tuple(_rows(family, ref, row, head, q_first, block_q)
                         for ref in (peak_ref, total_ref, anchor_ref, delta_ref, index_ref))

        def consumer(index):
            k_first = pair_first + index * block_k
            keys_view = (row, pl.ds(k_first, block_k), lax.div(head, heads // kv_heads))
            family.loaded(((k_ref.at[keys_view], k_smem.at[index], k_loaded.at[index]),
                           (v_ref.at[keys_view], v_smem.at[index], v_loaded.at[index])))
            lo, hi, whole = program.readers(k_first, q_steps, spans_of, row)

            def store_tiles(first, pieces):
                # Each tile's gradient entries straight to their place in the `[B, H, Qp, Kp]` outputs.
                for ref, piece in zip(tile_refs, pieces, strict=True):
                    ref[row, head, pl.ds(first, block_q), pl.ds(k_first, block_k)] = piece

            # The tiles before this block's first reader its partner reads.
            family.passed(streams, streamed[0], lo, first=streamed[0])
            adds = None if dq_ref is None else functools.partial(add_dq, dq_ref, index)
            coords_of = functools.partial(_coords, family, spans_of, row, k_first=k_first,
                                          shape=(block_q, block_k))
            params_of = functools.partial(_score_params, family, score_refs, kinds, row, head,
                                          k_first=k_first)
            step = functools.partial(_dkv_step, walk, (k_smem.at[index], v_smem.at[index]), rows_of,
                                     staged.at[index], coords_of, params_of, whole, adds, store_tiles,
                                     sums_smem.at[index])
            key_sums, (dk, dv) = family.walk(streams, ((step, lo, hi),),
                                             tuple(family.column_zeros(block_k) for _ in key_refs),
                                             ((block_k, width), (block_k, value_width)), walk.interpret,
                                             first=streamed[0])
            # The tiles past this block's last reader its partner reads.
            family.passed(streams, hi, streamed[1], first=streamed[0])
            out = (row, pl.ds(k_first, block_k), head)
            family.stored((dk, dv), (dk_ref.at[out], dv_ref.at[out]),
                          smem=out_smem.at[index, pl.ds(0, block_k)])
            if key_refs:
                # Each key gradient, the walk's column sums, straight to the `[B, H, Kp]` outputs.
                for ref, sums in zip(key_refs, key_sums, strict=True):
                    ref[row, head, pl.ds(k_first, block_k)] = sums
            if tile_refs:
                _unread_tiles(family, store_tiles, len(tile_refs), (block_q, block_k), (lo, hi), q_steps)

        def producer():
            family.stream(streams, views, *streamed)

        family.roles(consumers, consumer, producer)

    scoped, rings = _dkv_allocations(family, program.tiles, width, value_width, query.dtype,
                                     parts=_parts(walk.accurate, query.dtype), stages=stages)

    def entry(*refs):
        family.scoped(functools.partial(body, *refs), *scoped, streams=rings)

    gradient = jax.ShapeDtypeStruct((batch, keys_padded, heads, width), jnp.float32)
    grad_shapes = {Kind.KEY: jax.ShapeDtypeStruct((batch, heads, keys_padded), jnp.float32),
                   Kind.TILE: jax.ShapeDtypeStruct((batch, heads, queries, keys_padded), jnp.float32)}
    launch = family.kernel(
        entry, out_type=(gradient, gradient, *(grad_shapes[kind] for kind in graded)),
        grid=(heads, keys_padded // pair, batch), grid_names=("heads", "k_tiles", "batch"),
        consumers=consumers, interpret=walk.interpret, approx_math=walk.approx_math,
        name="attention_dkv_fused" if walk.fused else "attention_dkv",
        # The model's products this launch owns (`stated`): dK's, dV's and dP's, and dQ's where it adds it;
        # its `S` recomputed.
        cost=stated(walk.spec, batch, heads, walk.spec.queries or queries,
                    (2 if walk.fused else 1) * width + 2 * value_width, query, key, value, dout))
    operands = (query, key, value, dout, peak, total, anchor, delta, key_index, *leaves)
    if not walk.fused:
        return launch(*operands)
    # dQ added into zeros in place: one copy, or each consumer's own interpreted (`Family.added`).
    copies = consumers if walk.interpret else 1
    dq = jax.new_ref(jnp.zeros((copies, batch, queries, heads, width), jnp.float32))
    dk, dv = launch(*operands, dq)
    return jax.freeze(dq).sum(axis=0), dk, dv


def _launch_backward(query, key, value, dout, peak0, total0, initial_term, maximum, key_index, anchor_value,
                     shift, others, own, others_total, forward_total, start, end, query_ids, key_ids,
                     score_params, *, score_mod, score_kinds, score_dtypes, scale: float, prescale: float,
                     spec: algebra.Forward, accurate: bool, tiles: BackwardTiles, family: mosaic.Family,
                     interpret: bool):
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
    if walk.fused:
        dq, dk, dv = _dkv_launch(query, key, value, dout, maximum, forward_total, anchor, delta, chosen,
                                 spans, score_params, walk=walk)
        mass = algebra.split_mass(own, others_total, forward_total, shift)
        return (dq[..., :width], delta, anchor, maximum, forward_total, mass, chosen, dk[..., :width],
                dv[..., :value_width])
    # The dQ program reads its queries for their logits alone.
    dq_sum, keyed, rows = _dq_launch(by_power(query, prescale), key, value, dout, maximum, forward_total,
                                     anchor, delta, chosen, spans, score_params, walk=walk)
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
                                    spans, score_params, walk=walk)
    summed_kinds = [kind for kind in score_kinds if kind in DQ_GRADS]
    head_query = tuple(grad.sum(axis=-1)[None, :, :, None] if kind == Kind.HEAD else grad
                       for grad, kind in zip(dq_grads, summed_kinds, strict=True))
    return (jnp.swapaxes(dq, 1, 2)[..., :width], refined, anchor, maximum, denominator, mass, chosen,
            *head_query, dk[..., :width], dv[..., :value_width], *key_tile)
