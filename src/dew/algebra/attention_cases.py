"""Printable attention components, held off the runtime path (`dew.algebra.generate`, docs/design/algebra.md
§6.11): the forward's online step, walk factors and output, the split-KV merge, the backward's weights and
cotangent phases, and the masked softcap, sink and sliding-window hooks."""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

from dew.algebra.generate import Cases, ComponentForm, declare, float32_cases, normal_operands
from dew.nn import precision
from dew.nn.kernels.templates import attention, attention_algebra, attention_gradient
from dew.nn.kernels.templates.attention_backward_algebra import (
    derived_cotangents,
    derived_from_weights,
    derived_weights,
    softmax_gradients,
)


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
    state = attention_algebra.merge(attention_algebra.GENERATED_MERGE,
                                    attention_algebra.Partial(left, left_peak, left_total),
                                    attention_algebra.Partial(right, right_peak, right_total))
    return state.acc, state.m, state.total


def _split_kv_truth(left, left_peak, left_total, right, right_peak, right_total):
    floor = np.finfo(np.float32).min
    left_peak, right_peak = jnp.maximum(left_peak, floor), jnp.maximum(right_peak, floor)
    maximum = jnp.maximum(left_peak, right_peak)
    a, b = jnp.exp2(left_peak - maximum), jnp.exp2(right_peak - maximum)
    return a[:, None] * left + b[:, None] * right, maximum, a * left_total + b * right_total


def _initial_draw(seed, operands):
    initial, peak, mass, logits, values = normal_operands(seed, operands)
    mass = np.abs(mass) + 1
    # A wholly masked tile keeps the live state, and a fresh row can take a new maximum.
    logits[0] = -np.inf
    initial[1], peak[1], mass[1] = 0, np.finfo(peak.dtype).min, 0
    if seed % 2:
        logits[2] += 12
    return initial, peak, mass, logits, values


def _split_kv_draw(seed, operands):
    left, left_peak, left_total, right, right_peak, right_total = normal_operands(seed, operands)
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
    logits, dout, _, _ = normal_operands(seed, operands)
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


def _recompute_form(function, truth, *, keys_first, dropped):
    return ComponentForm(functools.partial(function, keys_first=keys_first),
                         functools.partial(truth, keys_first=keys_first),
                         _attention_recompute_cases(keys_first=keys_first, dropped=dropped))


KEYS_COTANGENT = declare(__name__, "keys_cotangent", ComponentForm(
    _keys_cotangent, _keys_truth,
    _backward_cases(((64, 32), (64, 32), (32,), (32,), (64, 32)), mask_count=1)))


QUERIES_COTANGENT = declare(__name__, "queries_cotangent", ComponentForm(
    _queries_cotangent, _queries_truth,
    _backward_cases(((32, 64), (32, 64), (32,), (32,), (32,), (32,), (32, 64), (32, 64)),
                    mask_count=2, total_at=3, accurate=True)))


ONLINE_STEP = declare(__name__, "online_softmax_step", ComponentForm(
    _online_step, _online_truth, dataclasses.replace(
        float32_cases(*[((q, d), (q,), (q,), (q, k), (k, d)) for q, k, d in ((32, 64, 64), (64, 64, 128))]),
        draw=_initial_draw)))


SPLIT_KV_MERGE = declare(__name__, "split_kv_merge", ComponentForm(
    _split_kv_merge, _split_kv_truth, dataclasses.replace(
        float32_cases(*[((q, d), (q,), (q,)) * 2 for q, d in ((32, 64), (64, 128))]), draw=_split_kv_draw)))


VALUES_GRADIENT = declare(__name__, "values_gradient", ComponentForm(
    _value_gradient, _value_gradient_truth, dataclasses.replace(
        float32_cases(*[((q, k), (q, v), (q,), (q,)) for q, k, v in ((32, 64, 128), (64, 128, 256))]),
        draw=_value_gradient_draw)))


QUERY_WEIGHTS = declare(__name__, "query_weights", _recompute_form(
    _backward_weights, _backward_weights_truth, keys_first=False, dropped=False))


KEY_WEIGHTS = declare(__name__, "key_weights", _recompute_form(
    _backward_weights, _backward_weights_truth, keys_first=True, dropped=False))


QUERY_DROPPED_COTANGENTS = declare(__name__, "query_dropped_cotangents", _recompute_form(
    _dropped_cotangents, _dropped_cotangents_truth, keys_first=False, dropped=True))


KEY_DROPPED_COTANGENTS = declare(__name__, "key_dropped_cotangents", _recompute_form(
    _dropped_cotangents, _dropped_cotangents_truth, keys_first=True, dropped=True))


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


SOFTCAP_TILE = declare(__name__, "softcap_tile", ComponentForm(_softcap_tile, _softcap_tile_truth, Cases(
    tuple((jax.ShapeDtypeStruct(shape, jnp.float32), jax.ShapeDtypeStruct(shape, jnp.bool_))
          for shape in ((32, 64), (64, 128))), _softcap_draw, seeds=8, policies=("default",))))


SINK_STATE = declare(__name__, "sink_state", ComponentForm(
    _sink_state, _sink_state_truth, float32_cases(((1, 32, 8),), ((2, 64, 16),))))


SLIDING_VISIBILITY = declare(__name__, "sliding_visibility", ComponentForm(
    _sliding_visibility, _sliding_visibility_truth, Cases(
        tuple(tuple(jax.ShapeDtypeStruct((n,), jnp.int32) for n in (keys, rows, rows, rows, keys))
              for rows, keys in ((32, 64), (64, 128))), _sliding_draw, seeds=8, policies=("default",))))


def _online_factors(logits, peak, *, keys_first):
    rows = peak.shape[0]
    keys = logits.shape[0 if keys_first else 1]
    acc = jnp.zeros((64, rows) if keys_first else (rows, 64), logits.dtype)
    values = jnp.zeros((keys, 64), logits.dtype)
    q = jnp.arange(rows)
    k = jnp.arange(keys)
    coords = attention_algebra.Coords(q[None, :] if keys_first else q[:, None],
                                      k[:, None] if keys_first else k[None, :], keys_first=keys_first)
    spec = attention_algebra.Forward("full", keys, ids=False, scale=0.5, natural=False)
    operand = attention_algebra.Operand(values.shape, values.dtype, lambda: values)
    tile = attention_algebra.Keys(coords, logits, operand, jnp.matmul)
    maximum, factor, weights, _ = attention_algebra._softmax(
        spec, attention_algebra.Partial(acc, peak, jnp.zeros_like(peak)), tile)
    return maximum, factor, weights


def _online_factors_truth(logits, peak, *, keys_first):
    scores = logits.T if keys_first else logits
    maximum = jnp.maximum(peak, scores.max(axis=1))
    weights = jnp.exp2(scores - maximum[:, None])
    return maximum, jnp.exp2(peak - maximum), weights.T if keys_first else weights


def _factor_draw(seed, operands, *, keys_first):
    logits, peak = normal_operands(seed, operands)
    scores = logits.T if keys_first else logits
    scores[0] = -np.inf
    peak[1] = np.finfo(np.float32).min
    if seed % 2:
        scores[2] += 12
    return logits, peak


def _factor_form(keys_first):
    tiles = []
    for rows, keys in ((32, 64), (64, 128)):
        scores = (keys, rows) if keys_first else (rows, keys)
        tiles.append((scores, (rows,)))
    draw = functools.partial(_factor_draw, keys_first=keys_first)
    return ComponentForm(functools.partial(_online_factors, keys_first=keys_first),
                         functools.partial(_online_factors_truth, keys_first=keys_first),
                         dataclasses.replace(float32_cases(*tiles), draw=draw))


def _walk_output(acc, peak, total):
    spec = attention_algebra.Forward("spans", 64, ids=False, scale=0.5, natural=False)
    return attention_algebra._derived_output(spec, attention_algebra.Partial(acc, peak, total))


def _walk_output_truth(acc, peak, total):
    mass = total[:, None]
    return jnp.where(mass == 0, 0, acc / jnp.where(mass == 0, 1, mass))


def _walk_output_draw(seed, operands):
    acc, peak, total = normal_operands(seed, operands)
    total = np.abs(total) + 1
    acc[0], total[0], peak[0] = 0, 0, np.finfo(np.float32).min
    if seed % 2:
        acc[1], total[1] = 2.0 ** 126, 2.0 ** 80
    return acc, peak, total


QUERY_WALK_FACTORS = declare(__name__, "query_walk_factors", _factor_form(keys_first=False))


KEY_WALK_FACTORS = declare(__name__, "key_walk_factors", _factor_form(keys_first=True))


WALK_OUTPUT = declare(__name__, "walk_output", ComponentForm(
    _walk_output, _walk_output_truth, dataclasses.replace(
        float32_cases(((32, 64), (32,), (32,)), ((64, 128), (64,), (64,))), draw=_walk_output_draw)))
