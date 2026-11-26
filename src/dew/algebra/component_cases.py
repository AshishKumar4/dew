"""Printable attention backward, recurrence and precision components, held off the runtime path."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dew.algebra.generate import Cases, ComponentForm, declare
from dew.nn import precision
from dew.nn.kernels.recurrence_algebra import derived_tile
from dew.nn.kernels.templates.attention_backward_algebra import derived_cotangents, derived_from_weights


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


def _pairs(products, decays, scales):
    return derived_tile("pairs").mapped({"products": products, "decays": decays, "scales": scales})


def _pairs_truth(products, decays, scales):
    return jnp.tril(products * decays * scales[:, None], k=-1)


def _keys_cotangent(weights, products, anchor, delta, seen):
    return derived_from_weights(keys_first=True, masked=True).mapped(
        dict(weights=weights, products=products, anchor=anchor, delta=delta, seen=seen))


def _keys_truth(weights, products, anchor, delta, seen):
    return jnp.where(seen, weights * (products - anchor[None, :] - delta[None, :]), 0)


def _queries_cotangent(logits, products, peak, total, anchor, delta, seen, at_anchor):
    return derived_cotangents(False, False, True, accurate=True).mapped(
        dict(logits=logits, products=products, peak=peak, total=total, anchor=anchor, delta=delta,
             seen=seen, at_anchor=at_anchor))


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
            values[total_at] = total
        return tuple(values)

    return Cases((operands,), draw, seeds=8, policies=("accurate" if accurate else "default",))


def _rounded_half(x):
    return precision._rounded(x, jnp.float16)


def _rounded_half_truth(x):
    return x.astype(jnp.float16).astype(x.dtype)


def _rounded(x):
    return precision._rounded(x, jnp.bfloat16)


def _rounded_truth(x):
    return x.astype(jnp.bfloat16).astype(x.dtype)


def _tangent(x, dx, value):
    return precision._rounded_tangent(jnp.bfloat16, (x,), (dx,), value)


def _tangent_truth(x, dx, value):
    return dx


SCORES = declare(__name__, "scores", ComponentForm(
    _scores, _scores_truth, _cases(*[((n, n), (n, n), (n,)) for n in (16, 32, 64)])))
ROUNDED = declare(__name__, "rounded", ComponentForm(_rounded, _rounded_truth, _cases(((16, 32),))))
TANGENT = declare(__name__, "rounded_tangent", ComponentForm(
    _tangent, _tangent_truth, _cases(((16, 32), (16, 32), (16, 32)))))
PAIRS = declare(__name__, "pairs", ComponentForm(_pairs, _pairs_truth, _cases(((32, 32), (32, 32), (32,)))))
ROUNDED_HALF = declare(__name__, "rounded_half", ComponentForm(
    _rounded_half, _rounded_half_truth, _cases(((16, 32),))))
KEYS_COTANGENT = declare(__name__, "keys_cotangent", ComponentForm(
    _keys_cotangent, _keys_truth, _backward_cases(((64, 32), (64, 32), (32,), (32,), (64, 32)), mask_count=1)))
QUERIES_COTANGENT = declare(__name__, "queries_cotangent", ComponentForm(
    _queries_cotangent, _queries_truth,
    _backward_cases(((32, 64), (32, 64), (32,), (32,), (32,), (32,), (32, 64), (32, 64)),
                    mask_count=2, total_at=3, accurate=True)))
