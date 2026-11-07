"""The attention template's plain-JAX mainloop (docs/design/kernel-templates.md §9): the algebra of
`attention_algebra` run over the same tiles, in the same order and with the same products as a kernel's
mainloop, in XLA ops, so it runs on any backend and is the boundary test's second arm
(tests/test_attention_algebra.py: the Triton mainloop, interpreted, and this one agree to the bit at the
same tiles). It takes a call at its launches' padded shapes, as `TritonForward` does."""

import dataclasses
import functools
from typing import Literal, overload

import jax
import jax.numpy as jnp

from dew.nn.kernels.templates import (
    attention_algebra as algebra,
    attention_gradient as gradient,
    attention_program,
    axes,
)
from dew.nn.kernels.templates.attention import (
    ACCURATE,
    DKV_GRADS,
    DQ_GRADS,
    NATURAL,
    Dropout,
    Gradients,
    Hooks,
    KeySpans,
    Kind,
    PaddedCall,
    Split,
    block_map,
    by_power,
    gradients,
    interpreted_dot,
    key_tiles,
    padded_call,
    prescaled,
    ranges_walk,
    shared_place,
    within,
)


def _tile_param(value: jax.Array, kind: Kind, row, head, rows: jax.Array, keys: jax.Array):
    """A score hook parameter's values for a tile of query rows `rows` and keys `keys` (index arrays) of
    batch row `row` and query head `head`, as `kind` lays them out (`Kind.axes`): a head's scalar, else the
    tile's `[block_q, block_k]` entries, a singleton along an axis it does not vary on."""
    if kind is Kind.OUTPUT:
        raise ValueError("an output hook's parameter has no score tile")
    if kind is not Kind.HEAD:
        row, head = shared_place(value.shape, row, head)
    places = {"batch": row, "heads": head, "queries": rows[:, None], "keys": keys[None, :]}
    return value[axes.named(kind.axes, places)]


def _output_param(value: jax.Array, kind: Kind, row, head, rows: jax.Array):
    """An output hook parameter's values for query rows `rows` of batch row `row` and query head `head`, as
    `kind` lays them out (`Kind.axes`): a head's scalar, else the output tile's `[block_q, Dv]` entries, a
    query's a singleton along the width."""
    if "keys" in kind.axes:
        raise ValueError("an output hook's parameter has no key axis")
    queries = rows if "width" in kind.axes else rows[:, None]
    return value[axes.placed(kind.axes, {"batch": row, "heads": head, "queries": queries})]


def _identity(step: jax.Array) -> jax.Array:
    """A walk's step as its key tile: a range's walk, one step a tile."""
    return step


def _composed(keys_at, tile_of):
    """`keys_at` of each step's key tile (`stepped`)."""
    return lambda step: keys_at(tile_of(step))


def _weighted(carry: algebra.Carry, *, rows) -> jax.Array:
    """A training walk's weighted values, its split sums joined (`algebra.joined`), where they overflow."""
    return algebra.joined(carry, rows).acc


def forward(spec: algebra.Forward, query: jax.Array, key: jax.Array, value: jax.Array, start: jax.Array,
            end: jax.Array, query_ids: jax.Array | None, key_ids: jax.Array | None, score_params,
            score_kinds: tuple[Kind, ...], output_params, output_kinds: tuple[Kind, ...], initial, *,
            block_q: int, block_k: int, output_mod=None, residuals: bool = True,
            drop: Dropout | None = None):
    """`TritonForward._fwd`'s outputs with its residuals from its operands (`query` `[B, Qp, H, Dp]`,
    `key` and `value` `[B, Kp, Hk, ...]`, the spans' rows and ids, the hooks' padded parameters, the
    initial state `(m0, total0, acc0)` or None), one query block and head at a time in sequence, its key
    tiles walked as the Triton mainloop walks them: the output, and with `residuals` its training step's
    (the split sums, `algebra.training_step`) maxima, denominators, anchor keys and split sums, else
    its plain step's output and None, so each rounds as the Triton launch it stands for."""
    batch, queries, heads, _ = query.shape
    keys, kv_heads, value_width = key.shape[1], key.shape[2], value.shape[-1]
    group = heads // kv_heads
    blocks = queries // block_q
    blocks_map = block_map(KeySpans(start, end, query_ids, key_ids, keys), block_q, block_k)
    ids = None if query_ids is None or key_ids is None else (query_ids, key_ids)

    def program(index):
        i, rest = index // (batch * heads), index % (batch * heads)
        row, head = rest // heads, rest % heads
        draw = None if drop is None else drop.draw(row, head)
        first_query = i * block_q
        rows = first_query + jnp.arange(block_q)
        q_pos = rows[:, None]
        q = jax.lax.dynamic_slice_in_dim(query[row, :, head], first_query, block_q)
        kv_head = head // group
        spans = (start[row, rows], end[row, rows]) if spec.form == "spans" else None
        row_ids = None if ids is None else (ids[0][row, rows], ids[1][row])

        def keys_at(form: algebra.Forward, j) -> algebra.Keys:
            # A key tile's logits in `form` (the call's, or the fold's rewalk's, `algebra.walked`).
            columns = j * block_k + jnp.arange(block_k)
            coords = algebra.Coords(q_pos, columns[None, :],
                                    None if spans is None else algebra.intervals(*spans),
                                    None if row_ids is None else (row_ids[0], row_ids[1][columns]), draw)
            params = tuple(_tile_param(param, kind, row, head, rows, columns)
                           for param, kind in zip(score_params, score_kinds, strict=True))
            k_tile = jax.lax.dynamic_slice_in_dim(key[row, :, kv_head], j * block_k, block_k)
            v_tile = jax.lax.dynamic_slice_in_dim(value[row, :, kv_head], j * block_k, block_k)
            return algebra.Keys(coords, algebra.modify(form, interpreted_dot(q, k_tile.T), coords, params),
                                algebra.Operand(v_tile.shape, v_tile.dtype, lambda: v_tile), interpreted_dot)

        def anchor_rows(keys):
            # Each row's value row at its key (the anchor's).
            return value[row, keys, kv_head]

        resumed = None if initial is None else (initial[0][row, head, rows], initial[1][row, head, rows],
                                                initial[2][row, head, rows])

        tile_of = _identity
        if spec.form == "spans" and blocks_map.ranges is not None:
            # Runs walk their ranges' tiles in turn.
            hi, tile_of = ranges_walk(blocks_map.ranges[row, i])
            lo = 0
        elif spec.form == "spans":
            lo, hi = blocks_map.lo[row, i], blocks_map.hi[row, i]
        else:
            lo, hi = key_tiles(spec.form, first_query, block_q, block_k, spec.keys)
        out_params = tuple(_output_param(param, kind, row, head, rows)
                           for param, kind in zip(output_params, output_kinds, strict=True))

        def walk(form: algebra.Forward, live):
            # The walk in `form`: the plain one where `live` is None, else the guarded one, those rows
            # shifting (`algebra.walked`).
            tiles_of = _composed(functools.partial(keys_at, form), tile_of)
            return attention_program.walk(form, lo, hi, tiles_of, anchor_rows, resumed, block_q, value_width,
                                          live=live, training=residuals, dropping=drop is not None)

        def finished(form: algebra.Forward, carry):
            # A walk's outputs in `form` (`attention_program.finished`), as the Triton mainloop's.
            tiles_of = _composed(functools.partial(keys_at, form), tile_of)
            out, *kept = attention_program.finished(form, carry, tiles_of, anchor_rows, (lo, hi), resumed,
                                                    training=residuals, output_mod=output_mod,
                                                    output_params=out_params)
            return (out.astype(query.dtype), *kept) if residuals else out.astype(query.dtype)

        # A training walk's weighted values are its split sums joined, where they overflow.
        weighted = functools.partial(_weighted, rows=anchor_rows) if residuals else None
        return algebra.walked(spec, walk, finished, weighted)

    mapped = jax.lax.map(program, jnp.arange(blocks * batch * heads))
    return _unmapped(mapped, (blocks, batch, heads, block_q, value_width), residuals=residuals)


def _unmapped(mapped, shape: tuple[int, ...], *, residuals: bool):
    """The programs' outputs `[blocks * batch * heads, bq, ...]` back to `[B, Qp, H, Dv]`, and with
    `residuals` theirs to `[B, H, Qp, ...]`, as `forward` returns them; `shape` the programs' grid and
    tile, `(blocks, batch, heads, block_q, value_width)`."""
    blocks, batch, heads, block_q, value_width = shape
    queries = blocks * block_q
    out = mapped[0] if residuals else mapped
    out = out.reshape(blocks, batch, heads, block_q, value_width).transpose(1, 0, 3, 2, 4)
    out = out.reshape(batch, queries, heads, value_width)
    if not residuals:
        return out, None
    _, m, denominator, anchors, others, own, others_total, shift = mapped

    def rowwise(x):
        return x.reshape(blocks, batch, heads, block_q).transpose(1, 2, 0, 3).reshape(batch, heads, queries)

    others = others.reshape(blocks, batch, heads, block_q, value_width).transpose(1, 2, 0, 3, 4)
    split = Split(others.reshape(batch, heads, queries, value_width), rowwise(own), rowwise(others_total),
                  rowwise(shift))
    return out, (rowwise(m), rowwise(denominator), rowwise(anchors), split)


@dataclasses.dataclass(frozen=True)
class Tiles:
    """The plain-JAX mainloop's config (`xla_mainloop`, `backward`): its walk's `block_q` by `block_k` tiles
    over a call padded as a backend whose largest tile is `pad` pads it (`padded_call`)."""

    block_q: int
    block_k: int
    pad: int


@overload
def xla_mainloop(config: Tiles, query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
                 hooks: Hooks | None = None, *, scale: float | None = None, residuals: Literal[False] = False,
                 natural: bool = NATURAL) -> jax.Array:
    ...


@overload
def xla_mainloop(config: Tiles, query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
                 hooks: Hooks | None = None, *, scale: float | None = None, residuals: Literal[True],
                 natural: bool = NATURAL) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Split]:
    ...


def xla_mainloop(config: Tiles, query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
                 hooks: Hooks | None = None, *, scale: float | None = None, residuals: bool = False,
                 natural: bool = NATURAL):
    """The template's forward on this mainloop at `config` (docs/design/kernel-templates.md's
    `xla_mainloop`): `query` `[B, Q, H, D]` over `key` and `value` `[B, K, Hk, D]` and `[B, K, Hk, Dv]` and
    the keys `spans` leaves each query, padded and walked at the config's tiles; with `residuals`, the
    running maxima and denominators `[B, H, Q]`, the anchor keys and the split sums too; `natural` keeps
    the logits natural (`algebra.Forward.base2`)."""
    hooks = Hooks() if hooks is None else hooks
    block_q, block_k = config.block_q, config.block_k
    # The query taken by its scale's power of two before its products (`prescale`).
    query, scale, power = prescaled(query, scale)
    call = padded_call(query, key, value, spans, hooks, config.pad)
    queries, value_width = call.queries, value.shape[-1]
    # Every form guards its sums where its values can overflow them (`algebra.Forward.headroom`), and the
    # fold its logits' reach (`algebra.walked`).
    spec = algebra.Forward(spans.form, spans.keys, call.query_ids is not None, scale, hooks.score_mod,
                           natural, algebra.headroom(value.dtype), hooks.initial is not None,
                           compensated=algebra.compensates(value.dtype), prescale=power)
    if not residuals:
        out, _ = padded_walk(spec, call, hooks, block_q=block_q, block_k=block_k, residuals=False)
        return out[:, :queries, :, :value_width]
    out, (m, denominator, anchors, split) = padded_walk(spec, call, hooks, block_q=block_q, block_k=block_k,
                                                        residuals=True)
    sums = Split(split.others[:, :, :queries, :value_width], split.own[:, :, :queries],
                 split.others_total[:, :, :queries], split.shift[:, :, :queries])
    return (out[:, :queries, :, :value_width], m[:, :, :queries], denominator[:, :, :queries],
            anchors[:, :, :queries], sums)


@overload
def padded_walk(spec: algebra.Forward, call: PaddedCall, hooks: Hooks, *, block_q: int, block_k: int,
                residuals: Literal[False]) -> tuple[jax.Array, None]:
    ...


@overload
def padded_walk(spec: algebra.Forward, call: PaddedCall, hooks: Hooks, *, block_q: int, block_k: int,
                residuals: Literal[True]) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, Split]]:
    ...


def padded_walk(spec: algebra.Forward, call: PaddedCall, hooks: Hooks, *, block_q: int, block_k: int,
                residuals: bool):
    """`xla_mainloop`'s walk of a padded call (`padded_call`) in `spec`'s form, which a caller holding a form
    of its own walks too (another mainloop's guarded rows, `attention_mosaic`): the output `[B, Qp, H, Dvp]`
    and, with `residuals`, the maxima, denominators, anchor keys and split sums at the padded shapes (else
    None)."""
    initial = None
    if hooks.initial is not None:
        state = hooks.initial
        widened = ((0, 0),) * 3 + ((0, call.value.shape[-1] - state.acc.shape[-1]),)
        initial = (call.rows(state.m, algebra.LEAST), call.rows(state.total),
                   jnp.pad(call.rows(state.acc), widened))
    return forward(
        spec, call.query, call.key, call.value, call.start, call.end, call.query_ids, call.key_ids,
        call.score_params, tuple(param.kind for param in hooks.score_params), call.output_params,
        tuple(param.kind for param in hooks.output_params), initial, block_q=block_q, block_k=block_k,
        output_mod=hooks.output_mod, residuals=residuals, drop=hooks.drop)


def _seen(spans: tuple[jax.Array, jax.Array, jax.Array | None, jax.Array | None], row, rows: jax.Array,
          columns: jax.Array) -> jax.Array:
    """The entries `[len(rows), len(columns)]` of batch row `row` its spans let each query see."""
    start, end, query_ids, key_ids = spans
    ids = (None if query_ids is None or key_ids is None
           else (query_ids[row, rows][:, None], key_ids[row, columns][None, :]))
    return within(columns[None, :], start[row, rows][:, None], end[row, rows][:, None], ids)


def _dq_launch(query, key, value, dout, peak, total, initial_term, maximum, key_index, anchor_value, shift,
               sums, spans, score_params, *, score_mod, score_kinds: tuple[Kind, ...], scale: float,
               prescale: float, grad_scale: float, how: algebra.Recompute, block_q: int, block_k: int,
               split: bool, accurate: bool, drop=None):
    """The Triton mainloop's dQ launch (`attention_triton._dq_launch`), a query block and head at a time:
    each query's denominator and `D - c` from the forward's split `sums` (`gradient.split_delta`), or without
    `split` from a first pass at the forward's maximum (`gradient.off_anchor_delta`) but for the rows whose
    forward walked them normalized (shift -1, `algebra.split_residuals`), which the first pass cannot
    recompute without the forward's denominator; under dropout (`drop`) from its own centred first pass
    (`gradient.dropout_centre`); then dQ and the dQ program's hook gradients under it, `D` refined once from
    its own sums (`gradient.refinement`)."""
    batch, queries, heads, _ = query.shape
    # The logits' scale as the forward took it: 1 for raw products, scaled in the weights (`how`).
    factor = scale if how.factor is None else 1.0
    keys, kv_heads = key.shape[1], key.shape[2]
    group = heads // kv_heads
    blocks = queries // block_q
    start, end, query_ids, key_ids = spans
    bounds = block_map(KeySpans(start, end, query_ids, key_ids, keys), block_q, block_k)
    summed = tuple(kind in DQ_GRADS for kind in score_kinds)

    def program(index):
        i, rest = index // (batch * heads), index % (batch * heads)
        row, head = rest // heads, rest % heads
        draw = None if drop is None else drop.draw(row, head)
        rows = i * block_q + jnp.arange(block_q)
        q_pos = rows[:, None]
        kv_head = head // group
        q = jax.lax.dynamic_slice_in_dim(query[row, :, head], i * block_q, block_q)
        do = jax.lax.dynamic_slice_in_dim(dout[row, :, head], i * block_q, block_q)
        peak0, total0, mean0 = peak[row, head, rows], total[row, head, rows], initial_term[row, head, rows]
        maxima, chosen = maximum[row, head, rows], key_index[row, head, rows]
        shifts = shift[row, head, rows]
        anchor_rows = jax.lax.dynamic_slice_in_dim(anchor_value[row, :, head], i * block_q, block_q)
        lo, hi, tile_of = bounds.lo[row, i], bounds.hi[row, i], _identity
        if bounds.ranges is not None:
            # Runs walk their ranges' tiles in turn, as the forward does.
            hi, tile_of = ranges_walk(bounds.ranges[row, i])
            lo = 0

        def formed(step, *, slopes: bool = False):
            # A key tile's coordinates, mask, keys and values, and its logits as the forward formed them,
            # with their hook's slopes where asked.
            j = tile_of(step)
            columns = j * block_k + jnp.arange(block_k)
            params = tuple(_tile_param(param, kind, row, head, rows, columns)
                           for param, kind in zip(score_params, score_kinds, strict=True))
            k_tile = jax.lax.dynamic_slice_in_dim(key[row, :, kv_head], j * block_k, block_k)
            v_tile = jax.lax.dynamic_slice_in_dim(value[row, :, kv_head], j * block_k, block_k)
            seen = _seen(spans, row, rows, columns)
            logits, slope = gradient.tile_logits(score_mod, interpreted_dot(q, k_tile.T), seen, q_pos,
                                                columns[None, :], params, scale=factor, slopes=slopes)
            return columns, seen, k_tile, v_tile, logits, slope

        before, state_total = gradient.off_anchor_start(peak0, total0, maxima, value.shape[-1], shifts)
        others, own, others_total, prior = (part[row, head, rows] for part in sums)
        m = maxima
        if draw is not None:
            # Dropout's D: each dropped product centred about its query's first visible tile's least one.
            def terms(j):
                columns, seen, _, v_tile, logits, _ = formed(j)
                products = gradient.dropped_products(interpreted_dot(do, v_tile.T),
                                                    draw(q_pos, columns[None, :]))
                return gradient.below(logits, m, how), products, seen, columns[None, :]

            anchor, delta, chosen = gradient.dropout_centre(lo, hi, terms, keys, block_q, state_total,
                                                           mean0)
        else:
            # The forward took the first pass (`gradient.split_delta`).
            anchor, delta, chosen = gradient.split_delta(others, own, others_total, prior, state_total,
                                                        anchor_rows, do, chosen, shifts)
            if not split:
                # D without the forward's split sums: the dQ launch's own pass over the keys.
                def logits_of(j):
                    columns, _, _, v_tile, logits, _ = formed(j)
                    return columns[None, :], logits, v_tile

                walked = attention_program.off_anchor_pass(lo, hi, logits_of, before, maxima, chosen, shifts,
                                                           how, interpreted_dot)
                first = gradient.off_anchor_delta(walked, anchor_rows, do, state_total, mean0,
                                                 key_index[row, head, rows], shifts)
                kept = shifts < 0
                prior = jnp.where(kept, prior, algebra.power2(walked.total, shifts))
                anchor, delta, chosen = (jnp.where(kept, ours, theirs)
                                         for ours, theirs in zip((anchor, delta, chosen), first, strict=True))

        def weighed(j):
            # A key tile's operands as the derived dQ walk reads them (`attention_program.dq_sums`).
            columns, seen, k_tile, v_tile, logits, slopes = formed(j, slopes=True)
            return columns[None, :], seen, k_tile, v_tile, logits, slopes

        found = attention_program.dq_sums(
            weighed, q, do, m, anchor, delta, prior, lo, hi, chosen,
            None if draw is None else functools.partial(draw, q_pos), hooked=score_mod is not None,
            score_kinds=score_kinds, summed=summed, prescale=prescale, grad_scale=grad_scale, how=how,
            accurate=accurate, dot=interpreted_dot)
        dq, delta, denominator, mass, grads = gradient.dq_finish(found, prior, state_total, peak0,
                                                                (total0, mean0), m, anchor, delta,
                                                                accurate=accurate)
        return dq, (delta, anchor, m, denominator, mass), chosen, _written(grads, score_kinds)

    dq, rows_out, chosen, written = jax.lax.map(program, jnp.arange(blocks * batch * heads))
    return _dq_outputs(dq, rows_out, chosen, written, score_kinds, blocks, batch, heads, queries)


def _written(grads, kinds: tuple[Kind, ...]) -> tuple[jax.Array, ...]:
    """A dQ program's hook gradients as it writes them (`DQ_GRADS`): a head's partial sum, a query's own."""
    return tuple(jnp.sum(grad)[None] if kind == Kind.HEAD else grad
                 for grad, kind in zip(grads, kinds, strict=True) if kind in DQ_GRADS)


def _dq_outputs(dq, rows_out, chosen, written, kinds: tuple[Kind, ...], blocks: int, batch: int, heads: int,
                queries: int):
    """The dQ programs' outputs `[blocks * batch * heads, bq, ...]` laid out as the Triton launch's."""
    def queried(x):
        # [blocks * batch * heads, bq, ...] to [B, H, Qp, ...].
        return jnp.moveaxis(x.reshape(blocks, batch, heads, *x.shape[1:]), 0, 2).reshape(
            batch, heads, queries, *x.shape[2:])

    grads, written = [], list(written)
    for kind in kinds:
        if kind not in DQ_GRADS:
            continue
        piece = written.pop(0)
        grads.append(piece.reshape(blocks, batch, heads, 1) if kind == Kind.HEAD else queried(piece))
    return jnp.swapaxes(queried(dq), 1, 2), tuple(queried(x) for x in rows_out), queried(chosen), grads


def _dkv_launch(query, key, value, dout, peak, total, anchor, delta, key_index, spans, score_params, *,
                score_mod, score_kinds: tuple[Kind, ...], scale: float, prescale: float, grad_scale: float,
                how: algebra.Recompute, block_q: int, block_k: int, accurate: bool, drop=None):
    """The Triton mainloop's dK/dV launch (`attention_triton._dkv_launch`), a key block and query head at a
    time, under the dQ launch's softmax at its tiles: the query head's dK, dV `[B, Kp, H, ...]` and the
    key and tile parameters' gradients (`DKV_GRADS`, `[B, H, Kp]` and `[B, H, Qp, Kp]`)."""

    batch, queries, heads, _ = query.shape
    # The logits' scale as the forward took it: 1 for raw products, scaled in the weights (`how`).
    factor = scale if how.factor is None else 1.0
    keys, kv_heads = key.shape[1], key.shape[2]
    group = heads // kv_heads
    blocks = keys // block_k
    start, end, query_ids, key_ids = spans
    ranges = block_map(KeySpans(start, end, query_ids, key_ids, keys), block_q, block_k)


    def program(index):
        j, rest = index // (batch * heads), index % (batch * heads)
        row, head = rest // heads, rest % heads
        draw = None if drop is None else drop.draw(row, head)
        columns = j * block_k + jnp.arange(block_k)
        kv_head = head // group
        k_tile = jax.lax.dynamic_slice_in_dim(key[row, :, kv_head], j * block_k, block_k)
        v_tile = jax.lax.dynamic_slice_in_dim(value[row, :, kv_head], j * block_k, block_k)

        def formed(i):
            # A query tile's operands as the derived dK/dV walk reads them (`attention_program.dkv_sums`):
            # its rows, queries, output cotangents, mask, logits as the forward formed them and their hook's
            # slopes, and dropout multipliers.
            rows = i * block_q + jnp.arange(block_q)
            q = jax.lax.dynamic_slice_in_dim(query[row, :, head], i * block_q, block_q)
            do = jax.lax.dynamic_slice_in_dim(dout[row, :, head], i * block_q, block_q)
            seen = _seen(spans, row, rows, columns)
            params = tuple(_tile_param(param, kind, row, head, rows, columns)
                           for param, kind in zip(score_params, score_kinds, strict=True))
            logits, slopes = gradient.tile_logits(score_mod, interpreted_dot(by_power(q, prescale), k_tile.T),
                                                 seen, rows[:, None], columns[None, :], params, scale=factor,
                                                 slopes=True)
            tiles = {"logits": logits, "q": q, "dout": do, "peak": peak[row, head, rows],
                     "total": total[row, head, rows], "anchor": anchor[row, head, rows],
                     "delta": delta[row, head, rows], "seen": seen,
                     "at_anchor": columns[None, :] == key_index[row, head, rows][:, None]}
            if slopes is not None:
                tiles["slope"] = slopes[0]
            if draw is not None:
                tiles["keep"] = draw(rows[:, None], columns[None, :])
            return tiles, slopes, seen

        def pulled(i, hooks, pieces):
            # Query tile `i`'s hook pieces (`gradient.pulled`): the tile parameters' gradients written at its
            # rows and the key parameters' summed.
            tiled, *grads = hooks
            tiled = tuple(jax.lax.dynamic_update_slice_in_dim(entries, piece, i * block_q, axis=0)
                          for entries, piece in zip(tiled, (piece for kind, piece in zip(score_kinds, pieces,
                                                                                       strict=True)
                                                            if kind == Kind.TILE), strict=True))
            return (tiled, *(grad + piece if kind == Kind.KEY else grad
                             for grad, kind, piece in zip(grads, score_kinds, pieces, strict=True)))

        tiled = tuple(jnp.zeros((queries, block_k), jnp.float32) for kind in score_kinds if kind == Kind.TILE)
        hooks = (tiled, *(jnp.zeros((block_k,) if kind == Kind.KEY else (), jnp.float32)
                          for kind in score_kinds))
        dk, dv, (tiled, *grads) = attention_program.dkv_sums(
            formed, k_tile, v_tile, ranges.q_lo[row, j], ranges.q_hi[row, j], pulled, hooks,
            hooked=score_mod is not None, dropped=draw is not None, grad_scale=grad_scale, how=how,
            accurate=accurate, score_kinds=score_kinds, dot=interpreted_dot)
        keyed = iter(grad for grad, kind in zip(grads, score_kinds, strict=True) if kind == Kind.KEY)
        columns_of = iter(tiled)
        return dk, dv, tuple(next(keyed) if kind == Kind.KEY else next(columns_of)
                             for kind in score_kinds if kind in DKV_GRADS)

    dk, dv, written = jax.lax.map(program, jnp.arange(blocks * batch * heads))
    return _dkv_laid_out(dk, dv, written, [kind for kind in score_kinds if kind in DKV_GRADS],
                         batch=batch, heads=heads, queries=queries, keys=keys, block_k=block_k)


def _dkv_laid_out(dk, dv, written, kinds, *, batch: int, heads: int, queries: int, keys: int, block_k: int):
    """`_dkv_launch`'s programs' results laid out: dK, dV and the key parameters' gradients (`kinds`, each of
    `written`'s) from `[blocks * batch * heads, bk, ...]` to `[B, Kp, H, ...]`, the tile parameters' from
    `[blocks * batch * heads, Qp, bk]` to `[B, H, Qp, Kp]`."""
    blocks = keys // block_k

    def keyed(x):
        return jnp.moveaxis(x.reshape(blocks, batch, heads, *x.shape[1:]), 0, 2).reshape(
            batch, heads, keys, *x.shape[2:]).swapaxes(1, 2)

    def tiles(x):
        return jnp.moveaxis(x.reshape(blocks, batch, heads, queries, block_k), 0, 3).reshape(
            batch, heads, queries, keys)

    return keyed(dk), keyed(dv), tuple(keyed(grad[..., None])[..., 0].swapaxes(1, 2) if kind == Kind.KEY
                                       else tiles(grad) for grad, kind in zip(written, kinds, strict=True))


def _launch(query, key, value, dout, peak, total, initial_term, maximum, key_index, anchor_value, shift,
            others, own, others_total, forward_total, start, end, query_ids, key_ids, score_params, *,
            score_mod, score_kinds: tuple[Kind, ...], score_dtypes: tuple[str, ...], scale: float,
            prescale: float, block_q: int, block_k: int, split: bool, natural: bool, accurate: bool,
            initial: bool, drop=None):
    """The Triton backward's outputs (`TritonBackward`), from this mainloop's two launches, recomputing the
    weights as a forward of the call's form computed them (`algebra.recompute`) from the call's own query
    taken by `prescale` (`attention.prescale`) for its logits alone, dQ and dK at the call's whole scale."""
    del score_dtypes
    spans = start, end, query_ids, key_ids
    how = algebra.recompute(algebra.Forward("full", 0, ids=False, scale=scale, score_mod=score_mod,
                                            natural=natural, initial=initial, prescale=prescale))
    shared = {"score_mod": score_mod, "score_kinds": score_kinds, "scale": scale, "prescale": prescale,
              "grad_scale": scale * prescale, "how": how,
              "block_q": block_q, "block_k": block_k, "accurate": accurate, "drop": drop}
    sums = others, own, others_total, forward_total
    # The dQ launch reads its queries for their logits alone.
    dq, softmax, chosen, dq_grads = _dq_launch(by_power(query, prescale), key, value, dout, peak, total,
                                               initial_term, maximum, key_index, anchor_value, shift, sums,
                                               spans, score_params, split=split, **shared)
    delta, anchor, m, denominator, _ = softmax
    dk, dv, key_grads = _dkv_launch(query, key, value, dout, m, denominator, anchor, delta, chosen, spans,
                                    score_params, **shared)
    return (dq, *softmax, chosen, *dq_grads, dk, dv, *key_grads)


def backward(config: Tiles, query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
             hooks: Hooks | None, dout: jax.Array, *,
             residuals: tuple[jax.Array, jax.Array, jax.Array, Split] | None = None, split: bool = True,
             scale: float | None = None, natural: bool = NATURAL, accurate: bool = ACCURATE,
             rounded: bool = True) -> Gradients:
    """The template's backward on this mainloop (`gradients`) at `config`, from the forward's `residuals`
    (this mainloop's `xla_mainloop`'s where None); with `split`, D from the forward's split sums, as the
    Triton mainloop's `backward`, `natural` keeping the forward's logits natural and `accurate` taking the
    backward's accurate form (`ACCURATE`)."""
    block_q, block_k, pad = config.block_q, config.block_k, config.pad
    if residuals is None:
        plain = Hooks() if hooks is None else Hooks(hooks.score_mod, hooks.score_params, hooks.initial,
                                                    drop=hooks.drop)
        _, maxima, totals, anchors, sums = xla_mainloop(config, query, key, value, spans, plain, scale=scale,
                                                        residuals=True, natural=natural)
        residuals = maxima, totals, anchors, sums
    initial = hooks is not None and hooks.initial is not None
    if hooks is not None and hooks.drop is not None and not split:
        raise ValueError("the first-pass D (split=False) takes no probability dropout; the split sums do")
    launch = functools.partial(_launch, block_q=block_q, block_k=block_k, split=split, natural=natural,
                               accurate=accurate, initial=initial, drop=None if hooks is None else hooks.drop)
    return gradients(launch, query, key, value, spans, hooks, dout, residuals, block=pad, scale=scale,
                     rounded=rounded)
