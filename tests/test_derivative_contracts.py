"""Float-range and nested-derivative contracts of Dew's custom-derivative seams (ledger row 2283, theme
G12): each seam's primal, its forward and reverse derivatives, a forward derivative of its gradient and a
second forward derivative, the last two again under a rematerialized call (`jax.checkpoint`), run at a
normal draw and at draws at float32's exponent edges. Each order runs (a custom derivative's tangent tree
and dtypes as JAX requires: rows 1861 and 1862, a boolean residual's tangent and a cotangent tree of the
wrong length), and each of its leaves is finite wherever the same order's float64 run is a normal float32
(row 1768, a contraction that overflowed before its division). Behaviour alone: no route, structure or bits
are read, and each seam is its public call."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import widened

from dew.nn.kernels import recurrence_chunk
from dew.sampling.flow import GaussianTransition

TINY, LARGEST = float(np.finfo(np.float32).tiny), float(np.finfo(np.float32).max)


def normal_float32(values) -> np.ndarray:
    """Whether each float64 value is a float32 a few roundings inside its normal range: from just above
    the least normal to just below the largest, or zero."""
    size = np.abs(np.asarray(values, np.float64))
    return (size == 0) | ((size >= TINY * (1 + 2.0**-16)) & (size <= LARGEST * (1 - 2.0**-16)))


def orders(fn, primals, direction, cotangent):
    """`fn` at each order the contract holds, by name: the primal, a forward and a reverse derivative, a
    forward derivative of the gradient of the output's sum against `cotangent` and a second forward
    derivative, the last two again with `fn` rematerialized."""
    def scalar(f):
        return lambda *x: jnp.sum(f(*x) * cotangent)

    def forward(f):
        return lambda *x: jax.jvp(f, x, direction)[1]

    def gradient(f):
        return jax.grad(scalar(f), argnums=tuple(range(len(primals))))

    remat = jax.checkpoint(fn)
    return {
        "primal": lambda: fn(*primals),
        "jvp": lambda: forward(fn)(*primals),
        "vjp": lambda: jax.vjp(fn, *primals)[1](cotangent),
        "jvp of grad": lambda: jax.jvp(gradient(fn), primals, direction)[1],
        "second jvp": lambda: jax.jvp(forward(fn), primals, direction)[1],
        "jvp of grad, remat": lambda: jax.jvp(gradient(remat), primals, direction)[1],
        "second jvp, remat": lambda: jax.jvp(forward(remat), primals, direction)[1],
    }


def assert_contract(fn, primals, direction, cotangent, label: str) -> None:
    """Every order of `fn` runs at the draw, and each leaf is finite wherever the same order's float64
    run is a normal float32."""
    narrow = orders(fn, primals, direction, cotangent)
    with jax.enable_x64(new_val=True):
        wide_primals, wide_direction, wide_cotangent = widened((primals, direction, cotangent))
        wide = orders(fn, wide_primals, wide_direction, wide_cotangent)
        truths = {name: jax.tree.map(np.asarray, order()) for name, order in wide.items()}
    for name, order in narrow.items():
        values = jax.tree.leaves(jax.tree.map(np.asarray, order()))
        for index, (value, truth) in enumerate(zip(values, jax.tree.leaves(truths[name]), strict=True)):
            if not np.issubdtype(value.dtype, np.inexact):
                continue
            held = normal_float32(truth) & np.isfinite(truth)
            broken = held & ~np.isfinite(value)
            assert not broken.any(), (f"{label}, {name}, leaf {index}: {int(broken.sum())} values not finite "
                                      f"where float64's are normal float32s, e.g. {value[broken][:3]}")


EDGES = {"normal": 1.0, "near the largest": 2.0**63, "near the least normal": 2.0**-63}
"""The draws' scales: unit values, and values whose products (a squared distance, a second derivative's
pairs) reach float32's largest or its least normal."""


@pytest.mark.parametrize("edge", EDGES)
def test_the_gaussian_kl_holds_every_order_at_float32s_edges(edge):
    """`GaussianTransition.kl` (sampling/flow.py's custom JVP and its pullback, row 1768): every order
    runs, and each is finite wherever float64 is a normal float32, at unit values and at values whose
    squared distance nears float32's largest or its least normal."""
    scale = EDGES[edge]
    rng = np.random.default_rng(1768)
    mean, reference = (jnp.asarray(rng.standard_normal((2, 3)) * scale, jnp.float32) for _ in range(2))
    variance = jnp.asarray(rng.uniform(0.5, 2.0, (2,)), jnp.float32)

    def kl(mean, reference, variance):
        return GaussianTransition(mean, variance).kl(reference)

    primals = (mean, reference, variance)
    direction = tuple(jnp.asarray(rng.standard_normal(np.shape(x)) * scale, jnp.float32) for x in primals)
    assert_contract(kl, primals, direction, jnp.ones((2,), jnp.float32), f"the Gaussian KL {edge}")


@pytest.mark.parametrize("packed", [False, True], ids=["one sequence", "packed"])
def test_the_chunked_delta_rule_holds_every_order(packed):
    """`recurrence_chunk.chunk_delta_rule` (the portable rule's custom derivative through
    `generation.linear_tangent`, rows 1861 and 1862): every order runs, its boolean document starts a
    residual with no tangent space, and each is finite wherever float64 is a normal float32."""
    keys = jax.random.split(jax.random.key(1861), 8)
    rows, length, heads, width = 1, 37, 2, 8
    query, key, value = (jax.random.normal(k, (rows, length, heads, width)) for k in keys[:3])
    g = -jax.nn.softplus(jax.random.normal(keys[3], (rows, length, heads, 1)))
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (rows, length, heads)))
    starts = jax.random.bernoulli(keys[5], 0.1, (rows, length)) if packed else None

    def rule(query, key, value, g, beta):
        return recurrence_chunk.chunk_delta_rule(query, key, value, g, beta, chunk_size=16, starts=starts)[0]

    primals = (query, key, value, g, beta)
    direction = tuple(jax.random.normal(keys[6], x.shape) for x in primals)
    cotangent = jax.random.normal(keys[7], query.shape)
    assert_contract(rule, primals, direction, cotangent, "the chunked delta rule")
