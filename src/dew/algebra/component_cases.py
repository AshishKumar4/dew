"""Printable attention, recurrence and precision components, held off the runtime path."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from dew.algebra.generate import Cases, ComponentForm, declare
from dew.nn import precision
from dew.nn.kernels import recurrence_algebra
from dew.nn.kernels.recurrence_algebra import apart, derived_tile
from dew.nn.kernels.templates import attention_algebra
from dew.nn.kernels.templates.attention_backward_algebra import (
    derived_cotangents,
    derived_from_weights,
    softmax_gradients,
)
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


def _rounded(x):
    return precision._rounded(x, jnp.bfloat16)


def _rounded_truth(x):
    return x.astype(jnp.bfloat16).astype(x.dtype)


def _tangent(x, dx, value):
    return precision._rounded_tangent(jnp.bfloat16, (x,), (dx,), value)


def _tangent_truth(x, dx, value):
    return dx


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


SCORES = declare(__name__, "scores", ComponentForm(
    _scores, _scores_truth, _cases(*[((n, n), (n, n), (n,)) for n in (16, 32, 64)])))
ROUNDED = declare(__name__, "rounded", ComponentForm(
    _rounded, _rounded_truth, _cases(((16, 32),), ((8, 64),))))
TANGENT = declare(__name__, "rounded_tangent", ComponentForm(
    _tangent, _tangent_truth, _cases(*[((m, n),) * 3 for m, n in ((16, 32), (8, 64))])))
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
