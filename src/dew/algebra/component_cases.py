"""Printable cases of recurrence and precision components, held off the runtime path."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dew.algebra.generate import Cases, ComponentForm, declare
from dew.nn import precision
from dew.nn.kernels.recurrence_algebra import derived_tile


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


def _rounded(x):
    return precision._rounded(x, jnp.bfloat16)


def _rounded_truth(x):
    return x.astype(jnp.bfloat16).astype(x.dtype)


def _tangent(x, dx, value):
    return precision._rounded_tangent(jnp.bfloat16, (x,), (dx,), value)


def _tangent_truth(x, dx, value):
    return dx


SCORES = declare(__name__, "scores", ComponentForm(
    _scores, _scores_truth, _cases(((16, 16), (16, 16), (16,)), ((32, 32), (32, 32), (32,)))))
ROUNDED = declare(__name__, "rounded", ComponentForm(_rounded, _rounded_truth, _cases(((16, 32),))))
TANGENT = declare(__name__, "rounded_tangent", ComponentForm(
    _tangent, _tangent_truth, _cases(((16, 32), (16, 32), (16, 32)))))
