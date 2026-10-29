"""Decode through the template's Mosaic mainloop (row 2079, link 1): a row's one query over its cache's keys
`[starts, lengths)`, each key head's query heads one tile's columns and its keys the tile's rows (`K Qᵀ`, as
the keys-first dK/dV holds them, FlashAttention 3's SdP_swapAB), so a step's products and softmax take the
group's columns (`DECODE_COLUMNS` at least) rather than wgmma's 64 rows. A program is one row, key head and
slice of the row's visible key tiles; its walk is the algebra's plain step on the tile held keys first
(`algebra.plain_step`), its weighted values `Oᵀ = Vᵀ Pᵀ` `[Dv, N]` in registers, the weights staged for the
product. Each slice's state goes out as it stands, and the slices join by their maxima in XLA
(`joined`)."""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import attention_algebra as algebra
from dew.nn.kernels.templates.attention import prescaled, whole
from dew.nn.kernels.templates.attention_mosaic_program import _coords, _views_of, layouts

DECODE_COLUMNS = 16
"""The narrowest tile a key head's query heads take, the group's heads first, the rest seeing no key: the
weights staged `[bk, N]` for `Vᵀ Pᵀ` take Mosaic GPU's 16-bit tiling (8, 16), so 16 columns, past wgmma's
least N of 8."""


class _Bounds(NamedTuple):
    """Each decode column's keys `[start, end)` `[B, N]` (`_coords`' spans: a column of no head sees none)."""

    start: jax.Array
    end: jax.Array
    query_ids: None = None
    key_ids: None = None


def decode(query: jax.Array, key: jax.Array, value: jax.Array, lengths: jax.Array, *, slices: int,
           starts: jax.Array | None = None, sinks: jax.Array | None = None, scale: float | None = None,
           natural: bool = True, block_k: int = 64, stages: int = 2, interpret: bool = False) -> jax.Array:
    """`query` `[B, 1, H, D]` over `key` `[B, K, Hk, D]` and `value` `[B, K, Hk, Dv]`, row b's keys
    `[starts[b], lengths[b])` (from 0 without `starts`), each query head's `sinks` logit in its softmax where
    given, in `slices` slices of each row's visible key tiles of `block_k` keys: `[B, 1, H, Dv]` in the
    query's dtype, as `attention.reference` over those keys gives it."""
    batch, _, heads, width = query.shape
    keys, kv_heads, value_width = key.shape[1], key.shape[2], value.shape[3]
    group = heads // kv_heads
    columns = whole(group, DECODE_COLUMNS)
    scale = width ** -0.5 if scale is None else scale
    query, scale, power = prescaled(query, scale)
    starts = jnp.zeros_like(lengths) if starts is None else starts
    starts, lengths = (jnp.asarray(bound, jnp.int32) for bound in (starts, lengths))
    # Each key head's query heads as its tile's columns, the rest empty.
    grouped = query[:, 0].reshape(batch, kv_heads, group, width)
    grouped = jnp.pad(grouped, ((0, 0), (0, 0), (0, columns - group), (0, 0)))
    padded = whole(keys, block_k)
    key, value = (jnp.pad(array, ((0, 0), (0, padded - keys), (0, 0), (0, 0))) for array in (key, value))
    real = jnp.arange(columns) < group
    bounds = _Bounds(jnp.where(real, starts[:, None], 0), jnp.where(real, lengths[:, None], 0))
    spec = algebra.Forward("spans", padded, ids=False, scale=scale, natural=natural,
                           initial=sinks is not None, prescale=power)
    initial = None
    if sinks is not None:
        # Each column's sink, its head's (none past the group), the first slice's initial state.
        initial = jnp.pad(jnp.asarray(sinks, jnp.float32).reshape(kv_heads, group),
                          ((0, 0), (0, columns - group)), constant_values=-jnp.inf)
    acc, m, total = _launch(grouped, key, value, starts, lengths, bounds, initial, spec=spec,
                            slices=slices, block_k=block_k, stages=stages, interpret=interpret)
    return joined(spec, acc, m, total, group).reshape(batch, 1, heads, value_width).astype(query.dtype)


def joined(spec: algebra.Forward, acc: jax.Array, m: jax.Array, total: jax.Array, group: int) -> jax.Array:
    """The slices' states (`acc` `[B, Hk, S, Dv, N]`, `m` and `total` `[B, Hk, S, N]`) joined into each
    head's output `[B, Hk, G, Dv]` fp32: each slice's sums taken to the slices' largest maximum
    (`algebra.carried`, its units the walk's), a row of no keys zero, as every attention call reads."""
    top = jnp.max(m, axis=2, keepdims=True)
    safe = jnp.where(top == -jnp.inf, 0.0, top)
    factor = jnp.where(m == -jnp.inf, 0.0, algebra.carried(spec, m, safe))
    mass = (total * factor).sum(axis=2)
    sums = (acc * factor[:, :, :, None, :]).sum(axis=2)
    out = jnp.where(mass[:, :, None, :] > 0, sums / jnp.where(mass > 0, mass, 1.0)[:, :, None, :], 0.0)
    return jnp.swapaxes(out, 2, 3)[:, :, :group]


def _launch(query, key, value, starts, lengths, bounds: _Bounds, initial, *, spec: algebra.Forward,
            slices: int, block_k: int, stages: int, interpret: bool):
    """The decode launch: each program's slice state, `acc` `[B, Hk, S, Dv, N]` fp32 (its weighted values
    keys first) and its maxima and denominators `[B, Hk, S, N]`, in the walk's units."""
    batch, kv_heads, columns, width = query.shape
    value_width = value.shape[3]
    family = layouts("sm90", query.dtype)[0]
    # One consumer, the rest of the residency the programs' own (C = 1, more slices: p2079 L1 v2).
    scoped = [plgpu.SMEM((columns, width), query.dtype), plgpu.SMEM((block_k, columns), value.dtype),
              plgpu.SMEM((value_width, columns), jnp.float32), plgpu.SMEM((columns,), jnp.float32),
              plgpu.Barrier(num_barriers=1)]
    rings = family.streams({"keys": ((block_k, width), key.dtype),
                            "values": ((block_k, value_width), value.dtype)}, stages, 1)
    leaves, treedef = jax.tree.flatten((starts, lengths, bounds, initial))

    def body(q_ref, k_ref, v_ref, *refs, streams):
        read, refs = refs[:len(leaves)], refs[len(leaves):]
        (acc_ref, rows_ref), refs = refs[:2], refs[2:]
        q_smem, p_smem, acc_smem, rows_smem, q_loaded = refs
        starts_of, lengths_of, bounds_of, initial_of = treedef.unflatten(read)
        row, kv_head, part = lax.axis_index("batch"), lax.axis_index("heads"), lax.axis_index("slices")
        views = _views_of({"keys": k_ref, "values": v_ref}, row, kv_head)
        streams = family.opened(streams, views)
        # The row's visible key tiles, cut into the slices' runs.
        first = lax.div(starts_of[row], block_k)
        last = pl.cdiv(lengths_of[row], block_k)
        per = pl.cdiv(jnp.maximum(last - first, 0), slices)
        lo = jnp.minimum(first + part * per, last)
        hi = jnp.minimum(lo + per, last)

        def consumer(index):
            del index
            family.loaded(((q_ref.at[row, kv_head], q_smem, q_loaded.at[0]),))
            shape = (block_k, columns)
            empty = algebra.Partial(family.zeros((value_width, columns)),
                                    family.column_full(columns, -jnp.inf), family.column_full(columns, 0.0))
            start = empty
            if initial_of is not None:
                sink = family.columns_of(initial_of.at[kv_head])
                resumed = algebra.resumed(spec, sink, jnp.where(sink == -jnp.inf, 0.0, 1.0), empty.acc)
                taken = part == 0
                start = algebra.Partial(empty.acc, jnp.where(taken, resumed.m, empty.m),
                                        jnp.where(taken, resumed.total, empty.total))

            def step(index, slots, carry, accumulators):
                k_first = index * block_k
                s = family.product(slots["keys"], q_smem.transpose((1, 0)), shape)
                slots.release("keys")
                coords = _coords(family, bounds_of, row, 0, k_first, shape, keys_first=True)
                logits, products = algebra.logits_of(spec, s, coords)

                def accumulate(acc, factor, parts, value):
                    ((weights, _),) = parts
                    staged = family.staged(weights, p_smem)
                    product = family.product(value.transpose((1, 0)), staged, (value_width, columns))
                    slots.release("values")
                    return factor(acc) + product

                keys = algebra.Keys(coords, logits, lambda: slots["values"], None, accumulate, products)
                return algebra.plain_step(spec, carry, keys), accumulators

            state, _ = family.walk(streams, ((step, lo, hi),), start, (), interpret, first=lo)
            family.stored((state.acc,), (acc_ref.at[row, kv_head, part],), smem=acc_smem)
            places = tuple(rows_ref.at[place, row, kv_head, part] for place in range(2))
            family.stored((state.m, state.total), places, smem=rows_smem)

        def producer():
            family.stream(streams, views, lo, hi)

        family.roles(1, consumer, producer)

    def entry(*refs):
        family.scoped(functools.partial(body, *refs), *scoped, streams=rings)

    out_type = (jax.ShapeDtypeStruct((batch, kv_heads, slices, value_width, columns), jnp.float32),
                jax.ShapeDtypeStruct((2, batch, kv_heads, slices, columns), jnp.float32))
    acc, rows = family.kernel(entry, out_type=out_type, grid=(kv_heads, slices, batch),
                              grid_names=("heads", "slices", "batch"), consumers=1, interpret=interpret,
                              name="attention_decode", reducing_warpgroups=family.warpgroups(1))(
                                  query, key, value, *leaves)
    return acc, rows[0], rows[1]
