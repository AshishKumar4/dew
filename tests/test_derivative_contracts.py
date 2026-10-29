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

from dew.nn import precision
from dew.nn.blocks import table_rows
from dew.nn.kernels import recurrence_chunk
from dew.nn.kernels.templates.gemm import template

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


def ran(order, label: str, name: str):
    """`order()`'s leaves as arrays, a failure to run named by the seam and the order that raised it."""
    try:
        return jax.tree.map(np.asarray, order())
    except Exception as error:
        raise AssertionError(f"{label}: its {name} does not run: {type(error).__name__}: {error}") from error


def assert_contract(fn, primals, direction, cotangent, label: str) -> None:
    """Every order of `fn` runs at the draw, and each leaf is finite wherever the same order's float64
    run is a normal float32."""
    narrow = orders(fn, primals, direction, cotangent)
    with jax.enable_x64(new_val=True):
        wide_primals, wide_direction, wide_cotangent = widened((primals, direction, cotangent))
        wide = orders(fn, wide_primals, wide_direction, wide_cotangent)
        truths = {name: ran(order, f"{label} in float64", name) for name, order in wide.items()}
    for name, order in narrow.items():
        values = jax.tree.leaves(ran(order, label, name))
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


def normal_draw(seed: int, *shapes, scale: float = 1.0) -> tuple:
    """Standard normal float32 arrays of `shapes` at `scale`, from `seed`."""
    rng = np.random.default_rng(seed)
    return tuple(jnp.asarray(rng.standard_normal(shape) * scale, jnp.float32) for shape in shapes)


@pytest.mark.parametrize("edge", EDGES)
def test_an_embedding_lookup_holds_every_order(edge):
    """`blocks.table_rows` (a gather whose reverse mode sums each token's cotangent into its row, row
    1869): every order runs in the table, the ids not differentiable, and each is finite wherever float64
    is a normal float32."""
    (table,) = normal_draw(1869, (11, 6), scale=EDGES[edge])
    ids = jnp.asarray([[0, 3, 3, 10], [7, 0, 2, 2]], jnp.int32)
    (direction,) = normal_draw(1870, table.shape, scale=EDGES[edge])
    (cotangent,) = normal_draw(1871, (2, 4, 6))
    assert_contract(lambda table: table_rows(table, ids, jnp.float32), (table,), (direction,), cotangent,
                    f"an embedding lookup {edge}")


@pytest.mark.parametrize("edge", EDGES)
def test_a_bf16_dot_holds_every_order(edge):
    """`precision.bf16_dot_general` (the BF16_BF16_F32 algorithm's dot and its cotangent's one rounding
    to bf16): every order runs, finite wherever float64 is a normal float32, at values whose products
    near float32's ends."""
    primals = normal_draw(956, (5, 8), (8, 3), scale=EDGES[edge])
    direction = normal_draw(957, (5, 8), (8, 3), scale=EDGES[edge])
    (cotangent,) = normal_draw(958, (5, 3))

    def dot(lhs, rhs):
        return precision.bf16_dot_general(lhs, rhs, (((1,), (0,)), ((), ())))

    assert_contract(dot, primals, direction, cotangent, f"a bf16 dot {edge}")


def test_a_wide_cpu_sum_of_products_holds_every_order(monkeypatch):
    """`precision.accumulated` in a process running wide CPU arithmetic (`WIDE_CPU_ARITHMETIC`, its own
    custom derivative of the float64-summed product): every order runs, finite wherever float64 is a
    normal float32."""
    monkeypatch.setenv(precision.WIDE_CPU_ARITHMETIC, "1")
    primals = normal_draw(343, (4, 16), (16, 5))
    direction = normal_draw(344, (4, 16), (16, 5))
    (cotangent,) = normal_draw(345, (4, 5))
    assert_contract(lambda lhs, rhs: precision.accumulated(jnp.matmul, lhs, rhs), primals, direction,
                    cotangent, "a wide CPU sum of products")


def test_a_grouped_gemm_holds_every_order():
    """The GEMM template's grouped product (`template.gemm`, its custom derivative through
    `generation.linear_tangent`): every order runs in the rows and the matrices, the group sizes not
    differentiable, finite wherever float64 is a normal float32."""
    sizes = jnp.asarray([3, 0, 5], jnp.int32)
    primals = normal_draw(146, (8, 16), (3, 16, 4))
    direction = normal_draw(147, (8, 16), (3, 16, 4))
    (cotangent,) = normal_draw(148, (8, 4))
    def grouped(rows, matrices):
        return template.gemm(rows, matrices, sizes, out_dtype=rows.dtype)

    assert_contract(grouped, primals, direction, cotangent, "a grouped GEMM")

