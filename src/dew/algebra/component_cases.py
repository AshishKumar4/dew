"""Printable attention, recurrence and precision components, held off the runtime path."""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
from dew.algebra.generate import Cases, ComponentForm, SurrogateBodies, SurrogateCase, declare

from dew.algebra import bits
from dew.nn import precision
from dew.nn.kernels import formats, recurrence_algebra, ssd
from dew.nn.kernels.recurrence_algebra import apart, derived_tile
from dew.nn.kernels.templates import attention, attention_algebra, attention_gradient
from dew.nn.kernels.templates.attention_backward_algebra import (
    derived_cotangents,
    derived_from_weights,
    derived_weights,
    softmax_gradients,
)
from dew.nn.kernels.templates.gemm import quantized
from dew.nn.kernels.templates.operands import Parts


def _arrays(seed, operands):
    rng = np.random.default_rng(seed)
    return tuple(np.asarray(rng.standard_normal(a.shape), dtype=a.dtype) for a in operands)


def _cases(*shapes):
    return Cases(tuple(tuple(jax.ShapeDtypeStruct(shape, jnp.float32) for shape in cell) for cell in shapes),
                 _arrays, seeds=8, policies=("default",))


def _scores(products, decays, scales):
    return derived_tile("scores").mapped({"products": products, "decays": decays, "scales": scales})


def _scores_truth(products, decays, scales):
    return jnp.tril(products * (decays * scales))


def _apart_truth(rows, rows_lo, columns, columns_lo):
    left = rows + rows_lo
    right = columns + columns_lo
    return left[:, None] - right[None, :]


def _apart_draw(seed, operands):
    rows, rows_lo, columns, columns_lo = _arrays(seed, operands)
    if seed == 1:
        # Rounding either compensated prefix before subtraction would lose every difference.
        rows.fill(2 ** 24)
        columns.fill(2 ** 24)
        rows_lo.fill(0.25)
        columns_lo.fill(0)
    return rows, rows_lo, columns, columns_lo


def _keys_cotangent(weights, products, anchor, delta, seen):
    return derived_from_weights(keys_first=True, masked=True).mapped(
        {"weights": weights, "products": products, "anchor": anchor, "delta": delta, "seen": seen})


def _keys_truth(weights, products, anchor, delta, seen):
    return jnp.where(seen, weights * (products - anchor[None, :] - delta[None, :]), 0)


def _queries_cotangent(logits, products, peak, total, anchor, delta, seen, at_anchor):
    return derived_cotangents(keys_first=False, dropped=False, masked=True, accurate=True).mapped(
        {"logits": logits, "products": products, "peak": peak, "total": total,
         "anchor": anchor, "delta": delta, "seen": seen, "at_anchor": at_anchor})


def _queries_truth(logits, products, peak, total, anchor, delta, seen, at_anchor):
    e = jnp.exp(jnp.minimum(logits - peak[:, None], 0))
    divisor = jnp.where(total == 0, 1, total)[:, None]
    centred = jnp.where(at_anchor, 0, products - anchor[:, None])
    return e, e / divisor, jnp.where(seen, e * ((centred - delta[:, None]) / divisor), 0)


def _backward_cases(shapes, *, mask_count, total_at=None, accurate=False):
    operands = tuple(jax.ShapeDtypeStruct(shape, jnp.bool_ if i >= len(shapes) - mask_count else jnp.float32)
                     for i, shape in enumerate(shapes))

    def draw(seed, operands):
        rng = np.random.default_rng(seed)
        values = [rng.uniform(-1, 1, a.shape).astype(a.dtype) if a.dtype != jnp.bool_
                  else rng.random(a.shape) > 0.25 for a in operands]
        if total_at is not None:
            total = np.abs(values[total_at]) + np.float32(0.25)
            if seed % 2 == 0:
                total[0] = 0
                values[-2][0] = False
            elif seed == 1:
                # The weight rounds to zero, but a large live dP keeps the cotangent normal.
                values[0][0], values[2][0], total[0] = 0, 87, 2.0 ** 25
                values[1][0, 1], values[4][0], values[5][0] = 3.8e30, 0, 0
                values[-2][0, 1], values[-1][0, 1] = True, False
            values[total_at] = total
        return tuple(values)

    return Cases((operands,), draw, seeds=8, policies=("accurate" if accurate else "default",))


def _rounded(x, *, dtype):
    return precision._rounded(x, dtype)


def _rounded_truth(x, *, dtype):
    return x.astype(dtype).astype(x.dtype)


def _tangent(x, dx, value, *, dtype):
    return precision._rounded_tangent(dtype, (x,), (dx,), value)


def _tangent_truth(x, dx, value):
    return dx


def _rounding_case(name, dtype):
    return SurrogateCase(name, ComponentForm(
        functools.partial(_rounded, dtype=dtype), functools.partial(_rounded_truth, dtype=dtype),
        _cases(((16, 32),), ((8, 64),))), {(): ComponentForm(
            functools.partial(_tangent, dtype=dtype), _tangent_truth,
            _cases(*[((m, n),) * 3 for m, n in ((16, 32), (8, 64))]))})


def _online_step(acc, peak, total, logits, values):
    rows, keys = logits.shape
    spec = attention_algebra.Forward("full", keys, ids=False, scale=0.5, natural=False)
    coords = attention_algebra.Coords(jnp.arange(rows)[:, None], jnp.arange(keys)[None, :])
    operand = attention_algebra.Operand(values.shape, values.dtype, lambda: values)

    def dot(a, b):
        return jnp.matmul(a, b, precision=precision.EXACT_FLOAT32)

    state = attention_algebra.plain_step(spec, attention_algebra.Partial(acc, peak, total),
                                         attention_algebra.Keys(coords, logits, operand, dot))
    return state.acc, state.m, state.total


def _online_truth(acc, peak, total, logits, values):
    maximum = jnp.maximum(peak, logits.max(axis=1))
    old = jnp.exp2(peak - maximum)
    weights = jnp.exp2(logits - maximum[:, None])
    return old[:, None] * acc + weights @ values, maximum, old * total + weights.sum(axis=1)


def _split_kv_merge(left, left_peak, left_total, right, right_peak, right_total):
    spec = attention_algebra.Forward("full", 2, ids=False, scale=0.5, natural=False)
    state = attention_algebra.merge(spec, attention_algebra.Partial(left, left_peak, left_total),
                                    attention_algebra.Partial(right, right_peak, right_total))
    return state.acc, state.m, state.total


def _split_kv_truth(left, left_peak, left_total, right, right_peak, right_total):
    floor = np.finfo(np.float32).min
    left_peak, right_peak = jnp.maximum(left_peak, floor), jnp.maximum(right_peak, floor)
    maximum = jnp.maximum(left_peak, right_peak)
    a, b = jnp.exp2(left_peak - maximum), jnp.exp2(right_peak - maximum)
    return a[:, None] * left + b[:, None] * right, maximum, a * left_total + b * right_total


def _initial_draw(seed, operands):
    initial, peak, mass, logits, values = _arrays(seed, operands)
    mass = np.abs(mass) + 1
    # A wholly masked tile keeps the live state, and a fresh row can take a new maximum.
    logits[0] = -np.inf
    initial[1], peak[1], mass[1] = 0, np.finfo(peak.dtype).min, 0
    if seed % 2:
        logits[2] += 12
    return initial, peak, mass, logits, values


def _split_kv_draw(seed, operands):
    left, left_peak, left_total, right, right_peak, right_total = _arrays(seed, operands)
    left_total, right_total = np.abs(left_total) + 1, np.abs(right_total) + 1
    # One empty slice on either side and an empty join. At large anchors the maxima must
    # remain separate from the masses; a scalar log-sum-exp would lose their ratio.
    empty = -np.inf if seed % 2 else np.finfo(left_peak.dtype).min
    left[0], left_peak[0], left_total[0] = 0, empty, 0
    right[1], right_peak[1], right_total[1] = 0, empty, 0
    left[2], left_peak[2], left_total[2] = 0, empty, 0
    right[2], right_peak[2], right_total[2] = 0, empty, 0
    left_peak[3], right_peak[3] = 2 ** 30, 2 ** 30
    return left, left_peak, left_total, right, right_peak, right_total


def _carry_product(name, left, right):
    del name
    left, right = (x.value() if isinstance(x, Parts) else x for x in (left, right))
    return jnp.matmul(left, right, precision=recurrence_algebra.STATE_PRECISION)


def _chunk_carry(state, keys, values, weights, kept):
    corrected = recurrence_algebra.corrected(values, lambda _: weights, [state], keys.dtype, _carry_product)
    (leaving,) = recurrence_algebra.carried([state], lambda _: keys, corrected, [kept], keys.dtype,
                                           _carry_product)
    return leaving, corrected


def _chunk_carry_truth(state, keys, values, weights, kept):
    corrected = values - weights @ state
    return kept * state + keys.T @ corrected, corrected


def _chunk_pullback(grad, keys, weights, queries, out_grad, local, kept):
    key_tile, weight_tile, query_tile = (Parts((a,), a.dtype) for a in (keys, weights, queries))
    (entered,), corrected = recurrence_algebra.key_reverse_carried(
        [grad], lambda _: key_tile, lambda _: weight_tile, lambda _: query_tile,
        Parts((out_grad,), out_grad.dtype), local, [kept], keys.dtype, _carry_product)
    return entered, corrected


def _chunk_pullback_truth(grad, keys, weights, queries, out_grad, local, kept):
    corrected = keys @ grad + local
    return kept * grad + queries.T @ out_grad - weights.T @ corrected, corrected


def _chunk_inputs(seed, operands):
    state, keys, values, weights, kept = _arrays(seed, operands)
    kept[:] = 2.0 ** -np.abs(np.round(kept))
    # Padding supplies no write or correction. A huge surviving memory needs its own decay.
    keys[0], values[0], weights[0] = 0, 0, 0
    if seed % 2:
        state[0], weights[:, 0], kept[0] = 2.0 ** 127, 0, 2.0 ** -126
    return state, keys, values, weights, kept


def _chunk_pullback_inputs(seed, operands):
    grad, keys, weights, queries, out_grad, local, kept = _arrays(seed, operands)
    kept[:] = 2.0 ** -np.abs(np.round(kept))
    keys[0], weights[0], queries[0], out_grad[0], local[0] = 0, 0, 0, 0, 0
    if seed % 2:
        grad[0], keys[:, 0], kept[0] = 2.0 ** 127, 0, 2.0 ** -126
    return grad, keys, weights, queries, out_grad, local, kept


def _value_gradient(logits, dout, peak, total):
    # The written backward's dV depends on no Q, K, V, score products or score cotangent.
    rows, keys = logits.shape
    zeros = jnp.zeros((rows,), logits.dtype)
    q = jnp.zeros((rows, 1), logits.dtype)
    k = jnp.zeros((keys, 1), logits.dtype)
    v = jnp.zeros((keys, dout.shape[-1]), dout.dtype)
    # This component models the device's float32 sum, not the suite's wider CPU-only reference.
    with precision.configured_precision(precision.FP32_SUMS):
        return softmax_gradients(logits, q, k, v, dout, peak, total, zeros, zeros, scale=1.0)[2]


def _value_gradient_truth(logits, dout, peak, total):
    used = jnp.isfinite(logits)
    exponent = jnp.where(used, logits - peak[:, None], -jnp.inf)
    weights = jnp.exp(exponent) / jnp.where(total > 0, total, 1)[:, None]
    return weights.T @ dout


def _value_gradient_draw(seed, operands):
    logits, dout, _, _ = _arrays(seed, operands)
    logits[0, :] = -np.inf
    logits[1, ::2] = -np.inf
    if seed % 2:
        logits[2, :] = np.float32(2.0 ** 30)
    peak = np.max(logits, axis=1)
    if seed % 2 == 0:
        peak[0] = np.finfo(logits.dtype).min
    safe = np.where(np.isfinite(peak), peak, np.inf)
    total = np.exp(logits.astype(np.float64) - safe[:, None]).sum(axis=1).astype(logits.dtype)
    return logits, dout, peak, total


def _chunk_level(system, inverse, level):
    rows, cols = recurrence_algebra.tile_indices(system.shape)
    return recurrence_algebra.doubled_level(-system, inverse, rows, cols, level, _carry_product)


def _chunk_level_truth(system, inverse, level):
    width = 1 << level
    groups = jnp.arange(system.shape[0]) // width
    join = (groups[:, None] % 2 == 1) & (groups[:, None] == groups[None, :] + 1)
    return inverse + jnp.where(join, inverse @ -system, 0) @ inverse


def _chunk_level_draw(seed, operands):
    size = operands[0].shape[0]
    system = _chunk_system(seed, size)
    level = 1 + seed % (size.bit_length() - 2)
    group = np.arange(size) // (1 << level)
    identity = np.eye(size, dtype=np.float64)
    diagonal = np.where(group[:, None] == group[None, :], system, 0)
    inverse = np.linalg.solve(diagonal + identity, identity).astype(system.dtype)
    return system, inverse, np.asarray(level, np.int32)


def _inverse_pullback(lower, cotangent):
    def issue(_, left, right):
        return jnp.matmul(left, right, precision=recurrence_algebra.STATE_PRECISION,
                          preferred_element_type=jnp.float32)

    return recurrence_algebra.key_system_back(cotangent, lower, lower.dtype, issue)


def _inverse_pullback_truth(lower, cotangent):
    inverse = jnp.eye(lower.shape[0], dtype=lower.dtype) + lower
    return jnp.tril(-(inverse.T @ jnp.tril(cotangent, -1)) @ inverse.T, -1)


def _inverse_pullback_draw(seed, operands):
    values, cotangent = _arrays(seed, operands)
    lower = np.tril(values / (2 * values.shape[0]), -1)
    if seed % 2:
        cotangent[np.triu_indices(values.shape[0], 1)] = np.nan
        np.fill_diagonal(cotangent, np.inf)
    else:
        lower[:2], lower[2:, :2] = 0, 0
        cotangent[:2], cotangent[2:, :2] = 0, 0
        cotangent[1, 0] = 2.0 ** 127
    return lower, cotangent


def _chunk_weighted(lower, values, strengths):
    return recurrence_algebra.weighted("apply_u", lower, values, strengths, _carry_product)


def _chunk_weighted_truth(lower, values, strengths):
    return (jnp.eye(lower.shape[0], dtype=lower.dtype) + lower) @ (values * strengths[:, None])


def _chunk_system(seed, size):
    rng = np.random.default_rng(seed)
    system = np.tril(rng.standard_normal((size, size)).astype(np.float32) / (2 * size), -1)
    # Split at a padded token: neither side may read across the block boundary.
    middle = system.shape[0] // 2
    system[middle:, :middle] = 0
    if seed % 2:
        # Isolate a top-binade edge: the inverse holds it without a product of two huge edges.
        system[:2], system[2:, :2] = 0, 0
        system[1, 0] = 2.0 ** 127
    return system


def _chunk_weighted_draw(seed, operands):
    lower, values, weights = _arrays(seed, operands)
    lower = np.tril(lower * 0.25, -1)
    weights = (np.abs(weights) * 0.25).astype(weights.dtype)
    weights[1] = 0
    if seed % 2:
        lower[1, 0], values[0], weights[0] = 2.0 ** 127, 0.125, 0.5
    return lower, values, weights


def _per_query(values, keys_first):
    return values[None, :] if keys_first else values[:, None]


def _backward_weights(logits, peak, total, *, keys_first):
    return derived_weights(keys_first=keys_first).mapped({"logits": logits, "peak": peak, "total": total})


def _backward_weights_truth(logits, peak, total, *, keys_first):
    maximum = _per_query(jnp.where(peak == -jnp.inf, jnp.inf, peak), keys_first)
    exponential = jnp.exp(logits - maximum)
    mass = _per_query(jnp.where(total == 0, 1.0, total), keys_first)
    return exponential, exponential / mass


def _dropped_cotangents(logits, products, peak, total, anchor, delta, keep, seen, at_anchor, *, keys_first):
    return derived_cotangents(keys_first=keys_first, dropped=True, masked=True, anchored=True).mapped(
        {"logits": logits, "products": products, "peak": peak, "total": total, "anchor": anchor,
         "delta": delta, "keep": keep, "seen": seen, "at_anchor": at_anchor})


def _dropped_cotangents_truth(logits, products, peak, total, anchor, delta, keep, seen, at_anchor, *,
                              keys_first):
    exponential, weights = _backward_weights_truth(logits, peak, total, keys_first=keys_first)
    centered = jnp.where(at_anchor, 0.0, products * keep - _per_query(anchor, keys_first))
    return exponential, weights, jnp.where(seen, weights * (centered - _per_query(delta, keys_first)), 0.0)


def _attention_recompute_cases(*, keys_first, dropped):
    def draw(seed, operands):
        rng = np.random.default_rng(seed)
        shape = operands[0].shape
        if keys_first:
            shape = shape[::-1]
        peak = rng.standard_normal((shape[0],)).astype(np.float32)
        logits = peak[:, None] - rng.uniform(0, 6, shape).astype(np.float32)
        total = rng.uniform(2, 4, (shape[0],)).astype(np.float32)
        # The empty row weighs zero. Plain fast recomputation does not clamp a logit that rounds
        # just above the forward's peak.
        logits[0], peak[0], total[0] = -np.inf, -np.inf, 0
        peak[1] = 1
        logits[1, 0] = np.nextafter(np.float32(1), np.float32(np.inf))

        def tiles(value):
            return value.T.copy() if keys_first else value

        if not dropped:
            return tiles(logits), peak, total
        products = rng.uniform(-1, 1, logits.shape).astype(np.float32)
        anchor = rng.uniform(-1, 1, peak.shape).astype(np.float32)
        delta = rng.uniform(-1, 1, peak.shape).astype(np.float32)
        keep = np.where(rng.random(logits.shape) < 0.25, 0, 1.25).astype(np.float32)
        seen = np.ones(logits.shape, dtype=np.bool_)
        seen[0], seen[:, -1] = False, False
        at_anchor = np.zeros(logits.shape, dtype=np.bool_)
        at_anchor[:, 0] = True
        if seed in (1, 3):
            # The unscaled centering overflows while the cotangent is finite. The same tile holds a
            # dropped key, an explicit anchor and a masked key, each read by a different branch.
            extreme = np.float32(2.0 ** 127)
            anchor[2], delta[2], total[2] = -extreme, 0, 4
            logits[2] = np.minimum(logits[2], 0)
            products[2, 1], keep[2, 1], logits[2, 1], peak[2] = extreme, 1, 0, 0
        return (tiles(logits), tiles(products), peak, total, anchor, delta, tiles(keep),
                tiles(seen), tiles(at_anchor))

    def of(rows, keys):
        tile = jax.ShapeDtypeStruct((keys, rows) if keys_first else (rows, keys), jnp.float32)
        query = jax.ShapeDtypeStruct((rows,), jnp.float32)
        mask = jax.ShapeDtypeStruct(tile.shape, jnp.bool_)
        if dropped:
            return tile, tile, query, query, query, query, tile, mask, mask
        return tile, query, query

    return Cases(tuple(of(rows, keys) for rows, keys in ((32, 64), (64, 128))), draw, seeds=8,
                 policies=("default",))


def _selective_decay_truth(step, rate, kept=None):
    decay = jnp.exp(step * rate)
    return decay if kept is None else jnp.where(kept != 0, decay, 0)


def _selective_decay_draw(seed, operands):
    rng = np.random.default_rng(seed)
    step = rng.uniform(0, 2, operands[0].shape).astype(np.float32)
    rate = np.asarray(-rng.uniform(0, 2, operands[1].shape), dtype=np.float32)
    if seed == 1:
        step[:] = 0
        rate[...] = -np.float32(3 * 2.0 ** 126)
    elif seed == 2:
        step[:] = np.float32(3 * 2.0 ** 126)
        rate[...] = -np.float32(2.0 ** -126)
    elif seed == 3:
        rate[...] = 0
    kept = () if len(operands) == 2 else ((rng.random(operands[2].shape) > 0.25).astype(np.float32),)
    return step, rate, *kept


def _key_memory_decay_truth(entered, leaving):
    return jnp.einsum("kv,kv->k", entered, leaving)


def _key_memory_decay_draw(seed, operands):
    entered, leaving = _arrays(seed, operands)
    if seed == 1:
        # A large carry and its reciprocal-scale cotangent produce ordinary finite terms.
        entered[0, :4], leaving[0, :4] = [2.0 ** 60, -2.0 ** 60, 2.0 ** 60, 0], 2.0 ** -60
    elif seed == 2:
        entering_mask = np.arange(entered.shape[0]) % 2 == 0
        entered[entering_mask] = 0
    return entered, leaving


def _recompute_form(function, truth, *, keys_first, dropped):
    return ComponentForm(functools.partial(function, keys_first=keys_first),
                         functools.partial(truth, keys_first=keys_first),
                         _attention_recompute_cases(keys_first=keys_first, dropped=dropped))

SCORES = declare(__name__, "scores", ComponentForm(
    _scores, _scores_truth, _cases(*[((n, n), (n, n), (n,)) for n in (16, 32, 64)])))
ROUNDED = SurrogateBodies(precision._rounded, precision._rounded_tangent, (1,), False, {
    (jnp.bfloat16,): _rounding_case("rounded", jnp.bfloat16),
    (jnp.float16,): _rounding_case("rounded_fp16", jnp.float16),
}).declared(__name__)
PREFIX_DIFFERENCE = declare(__name__, "prefix_difference", ComponentForm(
    apart, _apart_truth,
    dataclasses.replace(_cases(((32,), (32,), (64,), (64,))), draw=_apart_draw)))
KEYS_COTANGENT = declare(__name__, "keys_cotangent", ComponentForm(
    _keys_cotangent, _keys_truth,
    _backward_cases(((64, 32), (64, 32), (32,), (32,), (64, 32)), mask_count=1)))
QUERIES_COTANGENT = declare(__name__, "queries_cotangent", ComponentForm(
    _queries_cotangent, _queries_truth,
    _backward_cases(((32, 64), (32, 64), (32,), (32,), (32,), (32,), (32, 64), (32, 64)),
                    mask_count=2, total_at=3, accurate=True)))
ONLINE_STEP = declare(__name__, "online_softmax_step", ComponentForm(
    _online_step, _online_truth, dataclasses.replace(
        _cases(*[((q, d), (q,), (q,), (q, k), (k, d)) for q, k, d in ((32, 64, 64), (64, 64, 128))]),
        draw=_initial_draw)))
SPLIT_KV_MERGE = declare(__name__, "split_kv_merge", ComponentForm(
    _split_kv_merge, _split_kv_truth, dataclasses.replace(
        _cases(*[((q, d), (q,), (q,)) * 2 for q, d in ((32, 64), (64, 128))]), draw=_split_kv_draw)))
CHUNK_CARRY = declare(__name__, "chunk_carry", ComponentForm(
    _chunk_carry, _chunk_carry_truth, dataclasses.replace(
        _cases(*[((k, v), (c, k), (c, v), (c, k), (k, 1)) for c, k, v in ((16, 32, 32), (64, 64, 32))]),
        draw=_chunk_inputs)))
CHUNK_PULLBACK = declare(__name__, "chunk_pullback", ComponentForm(
    _chunk_pullback, _chunk_pullback_truth, dataclasses.replace(
        _cases(*[((k, v), (c, k), (c, k), (c, k), (c, v), (c, v), (k, 1))
                 for c, k, v in ((16, 32, 32), (64, 64, 32))]), draw=_chunk_pullback_inputs)))
CHUNK_SOLVE_LEVEL = declare(__name__, "chunk_solve_level", ComponentForm(
    _chunk_level, _chunk_level_truth, Cases(
        tuple((jax.ShapeDtypeStruct((c, c), jnp.float32), jax.ShapeDtypeStruct((c, c), jnp.float32),
               jax.ShapeDtypeStruct((), jnp.int32)) for c in (16, 32, 64)),
        _chunk_level_draw, seeds=8, policies=("default",))))
CHUNK_WEIGHTED = declare(__name__, "chunk_weighted", ComponentForm(
    _chunk_weighted, _chunk_weighted_truth,
    dataclasses.replace(_cases(*[((c, c), (c, v), (c,)) for c, v in ((16, 32), (32, 32), (64, 64))]),
                        draw=_chunk_weighted_draw)))
VALUES_GRADIENT = declare(__name__, "values_gradient", ComponentForm(
    _value_gradient, _value_gradient_truth, dataclasses.replace(
        _cases(*[((q, k), (q, v), (q,), (q,)) for q, k, v in ((32, 64, 128), (64, 128, 256))]),
        draw=_value_gradient_draw)))
CHUNK_INVERSE_PULLBACK = declare(__name__, "chunk_inverse_pullback", ComponentForm(
    _inverse_pullback, _inverse_pullback_truth,
    dataclasses.replace(_cases(*[((c, c), (c, c)) for c in (16, 32, 64)]), draw=_inverse_pullback_draw)))

QUERY_WEIGHTS = declare(__name__, "query_weights", _recompute_form(
    _backward_weights, _backward_weights_truth, keys_first=False, dropped=False))
KEY_WEIGHTS = declare(__name__, "key_weights", _recompute_form(
    _backward_weights, _backward_weights_truth, keys_first=True, dropped=False))
QUERY_DROPPED_COTANGENTS = declare(__name__, "query_dropped_cotangents", _recompute_form(
    _dropped_cotangents, _dropped_cotangents_truth, keys_first=False, dropped=True))
KEY_DROPPED_COTANGENTS = declare(__name__, "key_dropped_cotangents", _recompute_form(
    _dropped_cotangents, _dropped_cotangents_truth, keys_first=True, dropped=True))
SELECTIVE_DECAY = declare(__name__, "selective_decay", ComponentForm(
    recurrence_algebra.selective_decay, _selective_decay_truth, dataclasses.replace(
        _cases(((128,), (128,)), ((128,), (128,), ()), ((256,), (), (256,))), draw=_selective_decay_draw)))
KEY_MEMORY_DECAY = declare(__name__, "key_memory_decay", ComponentForm(
    recurrence_algebra.key_memory_decay, _key_memory_decay_truth, Cases(
        tuple((jax.ShapeDtypeStruct(shape, dtype), jax.ShapeDtypeStruct(shape, jnp.float32))
              for shape, dtype in (((64, 64), jnp.float32), ((128, 64), jnp.bfloat16))),
        _key_memory_decay_draw, seeds=8, policies=("default",))))


def _softcap_tile(logits, seen):
    capped, slopes = attention_gradient.tile_logits(attention.Capped(50.0), logits, seen, None, None, (),
                                                     scale=0.125, slopes=True)
    assert slopes is not None
    # Split out the negative-infinity sentinel for the finite error gate without losing any output.
    unseen = capped == -jnp.inf
    return jnp.where(unseen, 0.0, capped), slopes[0], unseen


def _softcap_tile_truth(logits, seen):
    z = logits / 400
    t = jnp.tanh(z)
    return jnp.where(seen, 50 * t, 0.0), 1 - t * t, ~seen


def _softcap_draw(seed, operands):
    rng = np.random.default_rng(seed)
    logits = rng.uniform(-200, 200, operands[0].shape).astype(np.float32)
    seen = rng.random(logits.shape) > 0.3
    logits[0, :5] = [0.0, -0.0, 1, -1, 1000]
    seen[0, :] = False
    logits[1, :3], seen[1, :3] = [0.0, -1.0, 1000], True
    return logits, seen


def _sink_state(values):
    initial = attention.sinks(values, values.shape[0], values.shape[1], 64)
    return initial.m, initial.total, initial.acc


def _sink_state_truth(values):
    peak = jnp.einsum("bqh->bhq", values)
    return peak, jnp.ones_like(peak), jnp.zeros((*peak.shape, 64), peak.dtype)


def _sliding_visibility(key, start, stop, queries, documents):
    return attention.within(key[None, :], start[:, None], stop[:, None],
                            (queries[:, None], documents[None, :]))


def _sliding_visibility_truth(key, start, stop, queries, documents):
    def visible(lo, hi, document):
        return (documents == document) & (key - lo >= 0) & (hi - key > 0)

    return jax.vmap(visible)(start, stop, queries)


def _sliding_draw(seed, operands):
    rng = np.random.default_rng(seed)
    rows, keys = operands[1].shape[0], operands[0].shape[0]
    key = np.arange(keys, dtype=np.int32) + 32
    stop = rng.integers(16, keys + 48, size=rows, dtype=np.int32)
    start = stop - rng.integers(0, 48, size=rows, dtype=np.int32)
    start[:3], stop[:3] = [32, 32, 40], [32, 33, 44]
    queries = rng.integers(0, 3, size=rows, dtype=np.int32)
    documents = (np.arange(keys, dtype=np.int32) // 8) % 3
    queries[:3] = [0, 0, 2]
    return key, start, stop, queries, documents


def _selective_adjoint_truth(cotangent, before, decay, written, step, value, b, dy, rate):
    carried = cotangent * decay * before
    write = cotangent * b
    return (step * write, rate * carried + value * write, (step * value) * cotangent,
            (before * decay + written) * dy, step * carried)


def _selective_adjoint_draw(seed, operands):
    rng = np.random.default_rng(seed)
    cotangent, before, decay, written, step, value, b, dy, rate = _arrays(seed, operands)
    decay = rng.uniform(0, 1, operands[2].shape).astype(np.float32)
    step = np.abs(step)
    rate = -np.abs(rate)
    # A reset contributes no entering-state derivative; a padded token has no write.
    decay[0], written[1], step[1] = 0, 0, 0
    return cotangent, before, decay, written, step, value, b, dy, rate


def _kda_strict_band(bands):
    return recurrence_algebra.block_masked(bands, strict=True, scale=1.0)


def _kda_band_truth(bands):
    sub = bands.shape[1]
    return jnp.tril(bands.reshape(-1, sub, sub), k=-1).reshape(bands.shape)


def _kda_band_draw(seed, operands):
    (bands,) = _arrays(seed, operands)
    # The inverse must never read the diagonal or an upper entry of a sub-chunk.
    token = np.arange(bands.shape[0]) % bands.shape[1]
    columns = np.arange(bands.shape[1])
    bands[columns[None, :] >= token[:, None]] = np.nan if seed % 2 else np.inf
    return (bands,)


def _gdn_values_truth(cotangent, values, beta):
    return jnp.einsum("cv,c->cv", cotangent, beta), jnp.einsum("cv,cv->c", cotangent, values)


def _gdn_values_draw(seed, operands):
    cotangent, values, beta = _arrays(seed, operands)
    beta = np.abs(beta) * np.float32(0.25)
    beta[0], values[1] = 0, 0
    return cotangent, values, beta


def _ssd_truth(x, b, c, a, state):
    def token(previous, inputs):
        value, write, read, decay = inputs
        next_state = jnp.exp(decay) * previous + value[:, None] * write[None, :]
        return next_state, next_state @ read

    final, out = jax.lax.scan(token, state, (x, b, c, a))
    return out, final


def _ssd_pullback_truth(x, b, c, a, state, dy, dnext):
    _, pull = jax.vjp(_ssd_truth, x, b, c, a, state)
    return pull((dy, dnext))


def _ssd_component_draw(seed, operands):
    vals = _arrays(seed, operands)
    x, b, c, a, state, *adjoints = vals
    a = -np.abs(a) * np.float32(0.125)
    a[0] = 0
    x[1], b[1] = 0, 0
    if seed == 1:
        a[len(a) // 2] = -32
    return x, b, c, a, state, *adjoints


def _ssd_cases(*, reverse):
    cases = []
    for chunk, channels, states in ((64, 32, 32), (64, 64, 32)):
        args = ((chunk, channels), (chunk, states), (chunk, states), (chunk,), (channels, states))
        cases.append((*args, args[0], args[-1]) if reverse else args)
    return dataclasses.replace(_cases(*cases), draw=_ssd_component_draw)


def _gemm_scaled_truth(partial, rows, columns):
    return partial * rows[:, None] * columns[None, :]


def _gemm_scaled(partial, rows, columns):
    return precision.scaled_partial(partial, rows[:, None], columns[None, :])


def _gemm_scale_draw(seed, operands):
    partial, rows, columns = _arrays(seed, operands)
    rows, columns = np.abs(rows), np.abs(columns)
    # Exercise row-first and column-first overflow avoidance as well as ordinary scales.
    if seed == 1:
        rows[0], columns[0], partial[0, 0] = 2.0 ** 127, 2.0 ** -80, 4
        partial[0, 1:] = 0
    elif seed == 2:
        columns[0], rows[0], partial[0, 0] = 2.0 ** 127, 2.0 ** -80, 4
        partial[1:, 0] = 0
    return partial, rows, columns


def _fp8_epilogue_truth(acc):
    # The recipe first stores its rounded activation, then scales each 128 columns independently.
    stored = acc.astype(jnp.bfloat16).astype(jnp.float64)
    blocks = stored.reshape(acc.shape[0], -1, 128)
    maximum = jnp.maximum(jnp.max(jnp.abs(blocks), axis=-1, keepdims=True), 1e-4)
    fraction, exponent = jnp.frexp(maximum)
    # The FP8 ceiling 448 is (7/8) × 2^9. Compare the exact fractions at a scale boundary.
    power = jnp.ldexp(jnp.ones_like(maximum), exponent - 9 + (fraction > 0.875))
    encoded = jnp.clip(blocks / power, -448, 448).astype(jnp.float8_e4m3fn).astype(jnp.float64)
    return stored, encoded.reshape(acc.shape), jnp.broadcast_to(power, blocks.shape).reshape(acc.shape)


def _fp8_epilogue_draw(seed, operands):
    (acc,) = _arrays(seed, operands)
    acc *= np.float32(4 ** (seed % 4))
    acc[0, :128] = 0
    acc[1, :4] = [448, -448, 224, -224]
    return (acc,)


SOFTCAP_TILE = declare(__name__, "softcap_tile", ComponentForm(_softcap_tile, _softcap_tile_truth, Cases(
    tuple((jax.ShapeDtypeStruct(shape, jnp.float32), jax.ShapeDtypeStruct(shape, jnp.bool_))
          for shape in ((32, 64), (64, 128))), _softcap_draw, seeds=8, policies=("default",))))
SINK_STATE = declare(__name__, "sink_state", ComponentForm(
    _sink_state, _sink_state_truth, _cases(((1, 32, 8),), ((2, 64, 16),))))
SLIDING_VISIBILITY = declare(__name__, "sliding_visibility", ComponentForm(
    _sliding_visibility, _sliding_visibility_truth, Cases(
        tuple(tuple(jax.ShapeDtypeStruct((n,), jnp.int32) for n in (keys, rows, rows, rows, keys))
              for rows, keys in ((32, 64), (64, 128))), _sliding_draw, seeds=8, policies=("default",))))
SELECTIVE_ADJOINT = declare(__name__, "selective_adjoint", ComponentForm(
    recurrence_algebra.selective_adjoint, _selective_adjoint_truth, dataclasses.replace(
        _cases(*[((n,),) * 8 + ((),) for n in (128, 256)]), draw=_selective_adjoint_draw)))
KDA_STRICT_BAND = declare(__name__, "kda_strict_band", ComponentForm(
    _kda_strict_band, _kda_band_truth, dataclasses.replace(
        _cases(((32, recurrence_algebra.SUB),), ((64, recurrence_algebra.SUB),)), draw=_kda_band_draw)))
GDN_VALUE_PULLBACK = declare(__name__, "gdn_value_pullback", ComponentForm(
    recurrence_algebra.gated_values_back, _gdn_values_truth, dataclasses.replace(
        _cases(*[((c, v), (c, v), (c,)) for c, v in ((32, 64), (64, 128))]), draw=_gdn_values_draw)))
SSD_CHUNK_FORWARD = declare(__name__, "ssd_chunk_forward", ComponentForm(
    ssd._chunk_forward, _ssd_truth, _ssd_cases(reverse=False)))
SSD_CHUNK_PULLBACK = declare(__name__, "ssd_chunk_pullback", ComponentForm(
    ssd._chunk_backward, _ssd_pullback_truth, _ssd_cases(reverse=True)))
GEMM_PARTIAL_SCALE = declare(__name__, "gemm_partial_scale", ComponentForm(
    _gemm_scaled, _gemm_scaled_truth, dataclasses.replace(
        _cases(*[((m, n), (m,), (n,)) for m, n in ((32, 128), (64, 128))]), draw=_gemm_scale_draw)))
FP8_EPILOGUE = declare(__name__, "fp8_epilogue", ComponentForm(
    quantized.blockwise_fp8_epilogue(), _fp8_epilogue_truth,
    dataclasses.replace(_cases(((32, 128),), ((64, 256),)), draw=_fp8_epilogue_draw)))


def _mxfp8_scale(raw):
    return formats.mxfp8_scale_bytes(jax.lax.bitcast_convert_type(raw, jnp.float32))


def _mx_scale_truth(value):
    _, e = jnp.frexp(value)
    exponent = jnp.clip(jnp.where(value == 0, -127, e - 9), -127, 127)
    finite = jnp.isfinite(value)
    return jnp.where(finite, exponent, 0), jnp.where(finite, exponent + 127, 255).astype(jnp.uint32)


def _mxfp8_scale_truth(raw):
    return _mx_scale_truth(jax.lax.bitcast_convert_type(raw, jnp.float32).astype(jnp.float64))


def _mx_scale_draw(seed, operands):
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 2 ** 31, operands[0].shape, dtype=np.uint32)
    raw.flat[:11] = [0, 1, 0x007fffff, 0x00800000, 0x047fffff, 0x04800000,
                     0x3f7fffff, 0x3f800000, 0x7f7fffff, 0x7f800000, 0x7fc00001]
    return (raw,)


def _mxfp8_epilogue(acc):
    codes, scales = quantized.quantize_mxfp8_epilogue()(acc)
    return jax.lax.bitcast_convert_type(codes, jnp.uint8), jax.lax.bitcast_convert_type(scales, jnp.uint8)


def _mxfp8_epilogue_truth(acc):
    blocks = acc.reshape(acc.shape[0], -1, 32)
    exponent, scale = _mx_scale_truth(jnp.max(jnp.abs(blocks), axis=-1, keepdims=True))
    narrowed = jnp.clip(jnp.ldexp(blocks, -exponent), -448, 448).astype(jnp.float8_e4m3fn)
    codes = jax.lax.bitcast_convert_type(narrowed, jnp.uint8)
    return codes.reshape(acc.shape), jnp.broadcast_to(scale, blocks.shape).astype(jnp.uint8).reshape(acc.shape)


def _mxfp8_draw(seed, operands):
    (acc,) = _arrays(seed, operands)
    acc *= np.float32(2.0 ** (seed * 20 - 80))
    acc[0, :32] = 0
    acc[1, :8] = [448, -448, 224, -224, 256, -256, 1, -0.0]
    return (acc,)


def _nvfp4_scale(codes):
    return jax.lax.bitcast_convert_type(bits.scales_decoded(codes), jnp.uint32)


def _nvfp4_scale_truth(codes):
    magnitude = codes.astype(jnp.uint32) & 127
    field, fraction = magnitude >> 3, magnitude & 7
    significand = fraction.astype(jnp.float64) * 0.125 + (field > 0)
    exact = jnp.ldexp(significand, jnp.maximum(field, 1).astype(jnp.int32) - 7)
    rounded = jax.lax.bitcast_convert_type(exact.astype(jnp.float32), jnp.uint32)
    return jnp.where(magnitude == 127, jnp.uint32(0x7fc00000), rounded) | (
        (codes.astype(jnp.uint32) & 0x80) << 24)


def _nvfp4_scale_draw(seed, operands):
    rows, columns = operands[0].shape
    codes = np.arange(rows * columns, dtype=np.uint32).astype(np.uint8)
    return (np.roll(codes, seed).reshape(rows, columns),)


def _nvfp4_payload(codes):
    return bits.nvfp4_codes(bits.packed(codes.T, bits.E2M1).T)


def _nvfp4_payload_truth(codes):
    values = jnp.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], jnp.float64)[codes & 7]
    return jnp.where(codes & 8 != 0, -values, values)


def _nvfp4_payload_draw(seed, operands):
    rng = np.random.default_rng(seed)
    (value,) = operands
    codes = rng.integers(0, 16, value.shape, dtype=np.uint32)
    codes[0, :16] = np.arange(16, dtype=np.uint32)
    return (codes,)


MXFP8_SCALE = declare(__name__, "mxfp8_scale", ComponentForm(_mxfp8_scale, _mxfp8_scale_truth, Cases(
    tuple((jax.ShapeDtypeStruct(shape, jnp.uint32),) for shape in ((32, 8), (64, 8))),
    _mx_scale_draw, seeds=8, policies=("default",))))
MXFP8_EPILOGUE = declare(__name__, "mxfp8_epilogue", ComponentForm(_mxfp8_epilogue, _mxfp8_epilogue_truth,
    dataclasses.replace(_cases(((32, 128),), ((64, 128),)), draw=_mxfp8_draw)))
NVFP4_SCALE = declare(__name__, "nvfp4_scale", ComponentForm(_nvfp4_scale, _nvfp4_scale_truth, Cases(
    tuple((jax.ShapeDtypeStruct(shape, jnp.uint8),) for shape in ((32, 8), (64, 8))),
    _nvfp4_scale_draw, seeds=8, policies=("default",))))
NVFP4_PAYLOAD = declare(__name__, "nvfp4_payload", ComponentForm(_nvfp4_payload, _nvfp4_payload_truth, Cases(
    tuple((jax.ShapeDtypeStruct(shape, jnp.uint32),) for shape in ((16, 32), (32, 64))),
    _nvfp4_payload_draw, seeds=8, policies=("default",))))
