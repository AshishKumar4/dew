"""The attention template's Mosaic GPU dQ launch (`attention_mosaic_backward`): each consumer's query tile
walked over the key tiles it reads, its dQ, and the row sums `dq_finish` refines between the two launches."""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from dew.nn.kernels import tuning
from dew.nn.kernels.templates import (
    attention_algebra as algebra,
    attention_gradient as gradient,
    mosaic,
    mosaic_sm100,
    outputs,
    placed,
    stores,
    swizzling,
)
from dew.nn.kernels.templates.attention import Dropout, Param
from dew.nn.kernels.templates.attention_mosaic_program import (
    BackwardTiles,
    TileWalk,
    _coords,
    _rows,
    _Spans,
    column_share,
    held_params,
    issued_into,
    modelled,
    param_types,
    placed_scratch,
    reducing,
    riding,
    scheduled_roles,
    score_streams,
    staged_after,
    stages_after_walks,
    stated,
    transposed_swizzles,
)
from dew.nn.kernels.templates.attention_mosaic_scores import (
    _Backward,
    _query_program,
    _query_tile_loaded,
    _relay_tiles,
    _relayed,
    _score_params,
    _scores,
)

DQ_ROWS = ("residual", "mass", "keyed_mass", "keyed_shift")
"""The row sums a dQ launch writes (`gradient.DqSums`' fields, `[B, H, Qp]` fp32 each, in the order of its
rows array), after them each summed dQ parameter's gradient (`DqSums.grads`) and then its weights' pullback
(`DqSums.weighed`): `dq_rows` lays a pass's sums out so, `dq_sums` reads them back."""


KEYED_ROWS = DQ_ROWS[2:]
"""The rows a launch of the keyed sums alone writes (`BackwardTiles.keyed_apart`): their mass and exponent,
`DQ_ROWS`' places in the dQ launch's rows, which `_dq_launch` takes them into."""

SUMS = ("dq", "keyed")
"""The sums a dQ launch may carry in its accumulators (`_dq_columns`' `walked`): dQ's and the accurate form's
keyed sums (`gradient.DqSums.keyed`), in this order."""


def _walks(walk: _Backward) -> tuple[tuple[str, ...], ...]:
    """The sums each of a dQ pass's launches carries (`SUMS`): dQ's alone in the fast form; dQ's and the keyed
    sums in the accurate form's one launch, or each in a launch of its own where its config keeps them apart
    (`BackwardTiles.keyed_apart`)."""
    if not walk.accurate:
        return (SUMS[:1],)
    return (SUMS[:1], SUMS[1:]) if walk.program.tiles.keyed_apart else (SUMS,)


def dq_rows(named: dict, grads: tuple, weighed: tuple) -> tuple:
    """A dQ pass's rows in the order its launch writes them (`DQ_ROWS`): its `named` rows, then its summed
    parameters' gradients and pullbacks."""
    return (*(named[name] for name in DQ_ROWS), *grads, *weighed)


def dq_sums(dq: jax.Array, keyed: jax.Array, rows, count: int) -> gradient.DqSums:
    """A dQ launch's sums from its dQ, its keyed sums and its `rows` (`dq_rows`' order), of `count` summed
    parameters."""
    named = dict(zip(DQ_ROWS, rows[:len(DQ_ROWS)], strict=True))
    params = tuple(rows[len(DQ_ROWS):])
    return gradient.DqSums(dq=dq, keyed=keyed, grads=params[:count], weighed=params[count:], **named)


def _dq_allocations(family: mosaic.Family, config: BackwardTiles, width: int, value_width: int, dtype,
                    stages: int, params: tuple = (), *, cotangents: bool = True) -> tuple[list, dict, list]:
    """The dQ program's allocations as `Family.scoped` takes them, in `_dq_launch`'s order, and those it
    scopes after its walks: its consumers' query and output-cotangent stages, their dQ output stage (fp32,
    its share of the config's `columns`; after the walks, in their slots, where the family scopes them apart,
    `stages_after_walks`) and their rows' stage, the two loads' barriers, each consumer's relay tile where
    its score cotangents reach dQ's product through shared memory (`Family.relay_bytes`, `_relayed`), and its
    key and value streams in `stages` slots (`Family.streams`), beside them the streams of the score hook's
    parameters that ride them (`params`, each its `Kind` and dtype: `score_streams`), staged plain. A launch
    of the keyed sums alone reads no `cotangents`: no output-cotangent stage, barrier or value stream."""
    consumers, block_q, block_k = config.consumers, config.block_q, config.block_k
    # A family placing its products on a role reads dQ out of tensor memory a stage's columns at a time
    # (`placed.Tcgen05Placed.stored`): its stage that many columns.
    out_width = width // config.columns
    if family.places_on_role:
        out_width = stores.staged_columns(out_width, (jnp.float32,))
    out = family.staged_memory((consumers, block_q, out_width), jnp.float32)
    after = [out] if stages_after_walks(family) else []
    read = (width, value_width) if cotangents else (width,)
    scoped = [*(family.staged_memory((consumers, block_q, columns), dtype) for columns in read),
              *([] if after else [out]),
              family.staged_memory((consumers, block_q), jnp.float32),
              *(family.load_barriers(consumers) for _ in read),
              *_relay_tiles(family, consumers, block_q, block_k, dtype)]
    blocks = {"keys": ((block_k, width), dtype), "values": ((block_k, value_width), dtype)}
    blocks = blocks if cotangents else {"keys": blocks["keys"]}
    hooked = score_streams(family, params, "keys", consumers * block_q, block_k)
    # The logits' and dP's products read the keys and values transposed.
    swizzles = {**transposed_swizzles(config, *blocks),
                **dict.fromkeys(hooked, swizzling.UNSWIZZLED),
                **(placed.walked_swizzles(QUERIES_FIRST, dtype, block_k) if family.places_on_role else {})}
    # A walk placed on a role holds a stream its ahead products alone read in one slot
    # (`placed.ahead_alone`), loaded last of a step's.
    once = ({name: blocks.pop(name) for name in placed.ahead_alone(QUERIES_FIRST)}
            if family.places_on_role else {})
    streams = family.streams({**blocks, **hooked}, stages, consumers, swizzles=swizzles)
    if once:
        streams = {**streams, **family.streams(once, 1, consumers, swizzles=swizzles)}
    return scoped, streams, after

def _dq_step_of(walk: _Backward, walked: tuple[str, ...], operands: tuple, stats: tuple, coords_of, params_of,
                whole, share, tile, *, placed_coords) -> Callable:
    """A dQ consumer's step carrying the sums `walked` (`_dq_step`), or a compute warpgroup's where its family
    places the products on a role (`_placed_dq_step`, its tiles' coordinates `placed_coords`), each step's
    key tile its index but where the walk is indexed (`tile`, `_Program.tile_of`)."""
    if walk.program.family.places_on_role:
        return functools.partial(_placed_dq_step, walk, stats, placed_coords, tile)
    return functools.partial(_dq_step, walk, walked, operands, stats, coords_of, params_of, whole, share,
                             tile)


def _dq_step(walk: _Backward, walked: tuple[str, ...], operands: tuple, stats: tuple, coords_of, params_of,
             whole, share, tile, index, slots, rows, accumulators):
    """A dQ consumer's key tile `index`: its scores and score cotangents from the key and value slots
    (`slots`) and its queries and output cotangents (`operands`) under its rows' statistics (`stats`), the
    tile's coordinates (`coords_of(k_first)`), its hook parameters (`params_of(like, k_first=k_first,
    slots=slots)`, those riding the keys' stream its slots') and
    whether its block sees it whole (`whole(index)`), the step's key tile its index but where the walk is
    indexed (`tile`, `_Program.tile_of`). The row sums are carried (`rows`); the products of the sums it
    carries (`walked`, `SUMS`: dQ's, the keyed sums') are issued into `accumulators` in place
    (`gradient.dq_issued`). A launch of the keyed sums alone reads no output cotangents
    (`BackwardTiles.keyed_apart`): its scores' alone, its cotangents formed of none and read by nothing, its
    rows the keyed sums' mass and exponent alone."""
    q, do = operands
    peak, prior, anchor, delta, chosen = stats
    family, tiles, empty = walk.program.family, walk.program.tiles, jnp.zeros((), jnp.float32)
    shape = (tiles.block_q, tiles.block_k)
    k_slot = slots["keys"]
    k_first = (index if tile is None else tile(index)) * tiles.block_k
    s = family.product(q, k_slot.transpose((1, 0)), shape)
    if "dq" in walked:
        dp = family.product(do, slots["values"].transpose((1, 0)), shape)
        slots.release("values")
    else:
        dp = family.zeros(shape)
    params = params_of(s, k_first=k_first, slots=slots)
    residual, mass, grads, weighed, keyed_mass, keyed_shift = rows
    # dQ's and the keyed sums' places stay empty: their products go to the accumulators.
    sums = gradient.DqSums(empty, residual, mass, grads, weighed, empty, keyed_mass, keyed_shift)

    def issued(*, masked: bool):
        # A tile every query sees whole (`masked` False) takes no mask; the row sums and the parts are taken
        # where the mask is read (`gradient.pulled`).
        # Its places tiles: vector broadcasts took dQ 1.77 to 2.06 ms under sinks at width 64 (ET-HELD-h100,
        # the config unchanged), its registers no fewer.
        coords = coords_of(k_first)
        scored = _scores(walk, coords, params, s, dp, peak, prior, anchor, delta, chosen, masked=masked)
        exponentials, _, cotangents, slopes, seen = scored
        terms, parts, keyed = gradient.dq_issued(sums, exponentials, cotangents, slopes, seen, k_slot.dtype,
                                                prior, scale=walk.grad_scale, kinds=walk.kinds,
                                                summed=walk.summed, accurate=walk.accurate, rows=True)
        keyed_rows = ((terms.keyed_mass, terms.keyed_shift) if "keyed" in walked
                      else (keyed_mass, keyed_shift))
        if "dq" not in walked:
            return keyed_rows, None, keyed
        row_sums = (terms.residual, terms.mass, terms.grads, terms.weighed, *keyed_rows)
        return row_sums, parts, keyed if "keyed" in walked else None

    # One walk, its tiles' masks chosen a tile at a time: a walk of two loops into one warpgroup accumulator,
    # the first empty, faults on an H100 under jax 0.11.2's Mosaic GPU lowering.
    rows, scaled, keyed = lax.cond(whole(index), functools.partial(issued, masked=False),
                                   functools.partial(issued, masked=True))
    accumulated = dict(zip(walked, accumulators, strict=True))
    # dQ's (and the keyed sums') products against the program's share of the keys' columns (`share`).
    k_columns = k_slot.at[:, pl.ds(*share)]
    if "dq" in walked:
        # dQ's parts formed past the choice, as each product is issued (`algebra.formed_parts`): the choice's
        # branches yield the scaled cotangents alone, not every part.
        parts = algebra.formed_parts(scaled, k_slot.dtype, accurate=walk.accurate, rows=True)
        accumulated["dq"] = issued_into(family, accumulated["dq"], parts, k_columns, walk.interpret)
    else:
        rows = (residual, mass, grads, weighed, *rows)
    if keyed is not None:
        accumulated["keyed"] = _keyed_issued(walk, accumulated["keyed"], keyed, k_columns, k_slot.dtype)
    return rows, tuple(accumulated[name] for name in walked)


def _keyed_issued(walk: _Backward, acc, keyed: tuple, k_columns, dtype):
    """The keyed sums `acc` after a tile's products (`gradient.dq_issued`'s `keyed`: their row factor, None
    where they take none, and the parts) against the keys' columns `k_columns`, of `dtype`."""
    family, (factor, parts) = walk.program.family, keyed
    if factor is None:
        return issued_into(family, acc, parts, k_columns, walk.interpret)
    if dtype == jnp.float16:
        # fp16's parts scaled (`product.Scaled`): the keyed sums rescaled by their row factor first, the
        # tile's products then restored into them (`issued_into`).
        acc = family.rescaled_products(acc, functools.partial(algebra.rescale, alpha=factor), (), k_columns,
                                       walk.interpret)
        return issued_into(family, acc, parts, k_columns, walk.interpret)
    # The keyed sums rescaled by their row factor, then the tile's products added in place.
    return family.rescaled_products(acc, functools.partial(algebra.rescale, alpha=factor), parts, k_columns,
                                    walk.interpret)

PART = 16
"""The keys a compute warpgroup of a placed queries-first walk takes of a step at once (`_placed_dq_step`):
a thread's query row's scores and their cotangents, 32 registers, beside its row's statistics."""

QUERIES_FIRST = (mosaic.Product("keys", placed.SCORES, "queries", transposed=True, ahead=True),
                 mosaic.Product("values", placed.COTANGENTS, "douts", transposed=True, ahead=True),
                 mosaic.Product("keys", placed.DQ, placed.COTANGENTS, accumulate=True))
"""A queries-first walk's products as a family placing them on a role issues them (`placed.walked_products`,
`placed.QUERIES`): its scores `S = Q Kᵀ` from the queries taken by the call's power and their cotangents
`dP = dO Vᵀ`, a step ahead, and dQ's onto its sums, its left the score cotangents the compute warpgroup
writes over the cotangents' turn (`_placed_dq_step`)."""


def _placed_dq_step(walk: _Backward, stats: tuple, coords_of, tile, index, slots, rows, accumulators):
    """The compute warpgroup's key tile `index` of a walk whose family places its products on a role queries
    first (`placed.QUERIES`, `QUERIES_FIRST`): the tile's scores and their cotangents its issuer's
    (`awaited`, a query a thread's row in tensor memory's own layout), `PART` keys at a time under its rows'
    statistics (`stats`) and coordinates (`coords_of(first key, shape=...)`), its key tile `tile(index)`
    where the walk is indexed (`_Program.tile_of`), the row sums carried (`rows`,
    `gradient.dq_issued`) and the scaled score cotangents written narrowed over the part's own columns of the
    cotangents' turn, read already, then given to the issuer for dQ's product (`given`). The plain call's
    alone (`MosaicBackward._walked_legal`)."""
    family, tiles, empty = placed.walked(walk.program.family), walk.program.tiles, jnp.zeros((), jnp.float32)
    peak, prior, anchor, delta, chosen = stats
    k_first = (index if tile is None else tile(index)) * tiles.block_k
    scores, cotangents = family.awaited(slots)
    pulled = family.turn_of(placed.COTANGENTS, slots.turn, narrowed=True)
    residual, mass, grads, weighed, keyed_mass, keyed_shift = rows
    for first in range(0, tiles.block_k, PART):
        s = family.read(scores, first=first, columns=PART, layout=mosaic_sm100.TMEM)
        dp = family.read(cotangents, first=first, columns=PART, layout=mosaic_sm100.TMEM)
        coords = coords_of(k_first + first, shape=(tiles.block_q, PART))
        exponentials, _, scored, slopes, seen = _scores(walk, coords, (), s, dp, peak, prior, anchor, delta,
                                                        chosen, masked=True)
        sums = gradient.DqSums(empty, residual, mass, grads, weighed, empty, keyed_mass, keyed_shift)
        terms, part, _ = gradient.dq_issued(sums, exponentials, scored, slopes, seen, pulled.dtype, prior,
                                            scale=walk.grad_scale, kinds=walk.kinds, summed=walk.summed,
                                            accurate=False, rows=True)
        residual, mass, grads, weighed = terms.residual, terms.mass, terms.grads, terms.weighed
        keyed_mass, keyed_shift = terms.keyed_mass, terms.keyed_shift
        (narrowed,) = algebra.formed_parts(part, pulled.dtype, accurate=False, rows=True)
        family.write(pulled, mosaic.unscaled_part(narrowed), first=first)
    family.given(slots)
    return (residual, mass, grads, weighed, keyed_mass, keyed_shift), accumulators


def _queries_first_issued(streams, reached: Callable, walked: placed.Tcgen05Walked, *, first, resident: tuple,
                          share: tuple) -> None:
    """A walk's issuer where it is placed queries first (`placed.Tcgen05Walked.issuing`): its products
    (`QUERIES_FIRST`) over the query tile's reach (`reached()`'s first two, formed in the issuer's role), a
    step ahead of the compute warpgroup that reads them, its resident queries and output cotangents
    (`resident`, their tiles and their loads' barriers) their lefts once in, dQ's `share` of its columns."""
    q_smem, do_smem, q_loaded, do_loaded = resident
    walked.issuing(streams, QUERIES_FIRST, *reached()[:2], first=first,
                   lefts=({"queries": q_smem.at[0], "douts": do_smem.at[0]},),
                   ready=(q_loaded.at[0], do_loaded.at[0]), shares={placed.DQ: share})


def _dq_launch(query, key, value, dout, peak, prior, anchor, delta, key_index, spans: _Spans | None,
               score_params: tuple, drop: Dropout | None, *, walk: _Backward):
    """`_dq_columns`' launches of the sums each carries (`_walks`), one of each of the config's `columns`, its
    share of the sums' columns, the sums joined along their widths: dQ, the keyed sums (None in the fast
    form) and the rows, the first's of the dQ launch, the keyed sums' mass and exponent their own launch's
    where they walk apart (`KEYED_ROWS`)."""
    columns = walk.program.tiles.columns
    width = query.shape[-1] // columns
    found, rows = {}, None
    for walked in _walks(walk):
        name = "attention_dq" + ("" if "dq" in walked else "_keyed")
        launched = [_dq_columns(query, key, value, dout, peak, prior, anchor, delta, key_index, spans,
                                score_params, drop, walk=walk, share=(column * width, width), walked=walked,
                                name=name + ("" if columns == 1 else f"_columns{column}"))
                    for column in range(columns)]
        for place, sum_name in enumerate(walked):
            found[sum_name] = (launched[0][0][place] if columns == 1
                               else jnp.concatenate([sums[place] for sums, _ in launched], axis=-1))
        at = DQ_ROWS.index(KEYED_ROWS[0])
        rows = launched[0][1] if rows is None else rows.at[at:at + len(KEYED_ROWS)].set(launched[0][1])
    return found["dq"], found.get("keyed"), rows

@tuning.lowered_once(("walk", "share", "name", "walked"))
def _dq_columns(query, key, value, dout, peak, prior, anchor, delta, key_index, spans: _Spans | None,
                score_params: tuple, drop: Dropout | None, *, walk: _Backward, share: tuple[int, int],
                name: str, walked: tuple[str, ...]):
    """The dQ program's sums over its key tiles (`gradient.dq_issued`), unfinished, those it carries
    (`walked`, `SUMS`): dQ's and, in the accurate form, the keys' product with the weights (`DqSums.keyed`),
    `[B, Qp, H, D]` fp32 each; and its rows (`DQ_ROWS`, `walk.summed`'s parameters after them; a launch of
    the keyed sums alone `KEYED_ROWS`), `[rows, B, H, Qp]` fp32, which `dq_finish` takes past the launch.
    Each query block of a pair is a consumer's, the key and value tiles streamed through a ring each (the
    keyed sums' launch alone no values). Its sums are its `share` of their columns (`(first, count)`).
    Returns its sums, in `walked`'s order, and the rows."""
    batch, queries, heads, width = query.shape
    kv_heads, value_width = key.shape[2], value.shape[-1]
    program, consumers, kinds, summed = walk.program, walk.consumers, walk.kinds, walk.summed
    family, block_q, block_k = program.family, program.tiles.block_q, program.tiles.block_k
    pair = consumers * block_q
    # A launch of the keyed sums alone reads no output cotangents (`BackwardTiles.keyed_apart`).
    cotangents = "dq" in walked
    rows_count, read = (len(DQ_ROWS) + 2 * sum(summed), 2) if cotangents else (len(KEYED_ROWS), 1)
    leaves, treedef = jax.tree.flatten((spans, score_params, drop))

    def body(*refs, streams, after=None):
        (q_ref, k_ref, v_ref, do_ref, peak_ref, prior_ref, anchor_ref, delta_ref, index_ref), refs = (
            refs[:9], refs[9:])
        (spans_of, score_refs, drop_of), refs = treedef.unflatten(refs[:len(leaves)]), refs[len(leaves):]
        sum_refs, rows_ref, refs = refs[:len(walked)], refs[len(walked)], refs[len(walked) + 1:]
        stages, refs = refs[:read], refs[read:]
        # Its output stage the program's where no scope after the walks holds it (`stages_after_walks`).
        out_smem, refs = (None, refs) if after is not None else (refs[0], refs[1:])
        row_smem, loaded, relay = refs[0], refs[1:1 + read], list(refs[1 + read:])
        q_smem, q_loaded = stages[0], loaded[0]
        # The output cotangents' stage and barrier, a launch reading them alone (`cotangents`).
        do_smem, do_loaded = (stages[1], loaded[1]) if cotangents else (None, None)
        row, head, views, streams, pair_first, streamed = _query_program(
            program, streams, k_ref, v_ref if cotangents else None, spans_of, score=(score_refs, kinds),
            heads=heads, kv_heads=kv_heads, queries=queries, consumers=consumers)

        def consumer(index):
            first = pair_first + index * block_q
            _query_tile_loaded(family, (q_ref, do_ref)[:read], stages, loaded, (row, head), first, index)
            stats = tuple(_rows(family, ref, row, head, first, block_q)
                          for ref in (peak_ref, prior_ref, anchor_ref, delta_ref, index_ref))
            lo, hi, whole = program.reach(first, spans_of, row)
            nothing = family.zeros((block_q,))
            operands = (family.held(q_smem.at[index]),
                        None if do_smem is None else family.held(do_smem.at[index]))
            shape = (block_q, block_k)
            # The tile's places formed a step, not held across the walk beside its sums (`_coords`).
            coords_of = functools.partial(_coords, family, spans_of, row, first, shape=shape,
                                          draw=None if drop_of is None else drop_of.draw(row, head))
            # The query tile's parameters held across the walk, read once (`held_params`), each tile's of the
            # keys' stream its step's (`_score_params`).
            params_of = functools.partial(
                _score_params, family, score_refs, kinds, row, head, first,
                held=held_params(family, score_refs, kinds, row, head, first, block_q),
                streamed=riding(family, kinds, "keys"), at=(index * block_q, 0))
            step = _dq_step_of(_relayed(walk, relay, index), walked, operands, stats, coords_of, params_of,
                               whole, share, program.tile_of(first, spans_of, row),
                               placed_coords=functools.partial(
                                   _coords, family, spans_of, row, first,
                                   draw=None if drop_of is None else drop_of.draw(row, head)))
            start = (nothing, nothing, (nothing,) * len(kinds), (nothing,) * len(kinds), nothing,
                     nothing + gradient.NO_SHIFT)
            # The tiles its partner reads around its own (`Family.walk_own`).
            (residual, mass, grads, weighed, keyed_mass, keyed_shift), accumulated = family.walk_own(
                streams, views, (lo, hi), streamed, index,
                lambda streams, lo, hi, counted: family.walk(
                    streams, ((step, lo, hi),), start, ((block_q, share[1]),) * len(walked),
                    walk.interpret, first=counted))
            rows = (row, pl.ds(first, block_q), head)
            written = [(acc, ref.at[rows]) for acc, ref in zip(accumulated, sum_refs, strict=True)]
            staged_after(after, out_smem, index,
                         lambda stage: family.stored(*zip(*written, strict=True), smem=stage))
            # Each row in a row's layout whatever its own (a call without a hook's keyed mass is a constant).
            summed_grads = tuple(part for part, on in zip(grads, summed, strict=True) if on)
            summed_weighed = tuple(part for part, on in zip(weighed, summed, strict=True) if on)
            parts = (dq_rows({"residual": residual, "mass": mass, "keyed_mass": keyed_mass,
                              "keyed_shift": keyed_shift}, summed_grads, summed_weighed) if cotangents
                     else (keyed_mass, keyed_shift))
            family.stored(tuple(family.as_row(part) for part in parts),
                          tuple(rows_ref.at[place, row, head, pl.ds(first, block_q)]
                                for place in range(rows_count)),
                          smem=row_smem.at[index])

        def producer():
            family.stream(streams, views, *streamed)

        scheduled_roles(family, consumers, consumer, producer, functools.partial(
            _queries_first_issued, streams, functools.partial(program.reach, pair_first, spans_of, row),
            first=streamed[0], resident=(q_smem, do_smem, q_loaded, do_loaded), share=share))

    scoped, rings, after = _dq_allocations(family, program.tiles, width, value_width, query.dtype,
                                           program.stages,
                                           params=param_types(map(Param, score_params, kinds)),
                                           cotangents=cotangents)

    products = placed_scratch(family, block_k, query.dtype, (), (share[1],))

    def entry(*refs):
        family.scoped(functools.partial(body, *refs), *scoped, streams=rings, after_walks=tuple(after),
                      **products)

    # Its outputs as its query's axes (`outputs.Output`): dQ's share of the columns, and its rows' statistics.
    rows, positions, heads_of = (outputs.axis(query, index) for index in range(3))
    sums = outputs.output(jnp.float32, rows, positions, heads_of, column_share(query, share[1]))
    stats_shape = outputs.output(jnp.float32, rows_count, rows, heads_of, positions)
    out = family.kernel(
        entry, out_type=(*(sums,) * len(walked), stats_shape),
        grid=(heads, queries // pair, batch), grid_names=("heads", "q_tiles", "batch"), consumers=consumers,
        interpret=walk.interpret, approx_math=walk.approx_math,
        name=name,
        reducing_warpgroups=reducing(family, consumers, walk.reduces),
        registers=walk.registers(dq=True),
        # Its walk's statement, its work its emission's (`stated`); its share of dQ's product the model's it
        # owns (`modelled`), the keyed sums' none.
        cost=stated(walk.spec, batch, heads, query, key, *((value, dout) if cotangents else ()),
                    tile_walk=TileWalk((queries, key.shape[1]), (block_q, block_k), consumers,
                                       common=family.feeds_itself)),
        model_flops=modelled(walk.spec, batch, heads, walk.spec.queries or queries,
                             share[1] if cotangents else 0))(
            query, key, value, dout, peak, prior, anchor, delta, key_index, *leaves)
    return tuple(out[:-1]), out[-1]
