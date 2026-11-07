"""A CPU's float32 dense products summed blocked and pairwise (`dew.nn.pairwise`, row 4863): the block each
contraction takes from the CPU's facts, the products a dense layer forms taking it in fixed float32 sums,
and its error against float64, forward and both gradients, below XLA's one-chain sum and within a
vectorized dot's."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew.nn import pairwise, precision


def test_a_contractions_block_keeps_its_rounding_path_within_a_vectorized_dots():
    """`contraction_block`: a power of two no longer than the depth whose path (its chain and the tree over
    its blocks) is no longer than a vectorized dot's over the CPU's lanes times its FMAs in flight: 2 at
    64 and 128 deep with AVX-512's 16 lanes, 16 at 1024, 64 at 4096; narrower vectors take longer blocks,
    and a depth one block holds is the plain dot's."""
    assert [pairwise.contraction_block(depth, 16) for depth in (64, 128, 1024, 4096)] == [2, 2, 16, 64]
    for lanes in (4, 8, 16):
        for depth in (3, 17, 96, 2880, 14336):
            block = pairwise.contraction_block(depth, lanes)
            assert block & (block - 1) == 0 and 1 <= block <= depth
    assert pairwise.contraction_block(4096, 8) >= pairwise.contraction_block(4096, 16)
    assert pairwise.vector_lanes() in (4, 8, 16)


def test_a_pairwise_product_takes_every_dot_generals_axes_and_bounds_its_partials():
    """`pairwise_dot_general` of batch axes, two contracted axes in another order than the operands hold
    them, and free axes on both sides: the dot's own result where every sum is exact (integers), in its
    order; empty rows, columns or contraction, and their gradients, the dot's zeros; and a product whose
    blocks' sums pass the workspace (`workspace_bytes`) sums the same bits a run of columns at a time as
    whole."""
    rng = np.random.default_rng(4863)
    lhs = jnp.asarray(rng.integers(-3, 4, size=(2, 5, 24, 3, 7)), jnp.float32)
    rhs = jnp.asarray(rng.integers(-3, 4, size=(7, 2, 24, 4)), jnp.float32)
    numbers = (((4, 2), (0, 2)), ((0,), (1,)))
    np.testing.assert_array_equal(np.asarray(pairwise.pairwise_dot_general(lhs, rhs, numbers)),
                                  np.asarray(jax.lax.dot_general(lhs, rhs, numbers)))
    for shapes in (((4, 0), (0, 5)), ((0, 6), (6, 5)), ((4, 6), (6, 0))):
        empty = [jnp.zeros(shape, jnp.float32) for shape in shapes]
        product = functools.partial(pairwise.pairwise_dot_general, dimension_numbers=(((1,), (0,)), ((), ())))
        value, pull = jax.vjp(product, *empty)
        assert value.shape == (shapes[0][0], shapes[1][1]) and not np.any(np.asarray(value))
        assert [gradient.shape for gradient in pull(jnp.ones_like(value))] == [tuple(s) for s in shapes]
    a = jnp.asarray(rng.standard_normal((1, 8, 1024)), jnp.float32)
    b = jnp.asarray(rng.standard_normal((1, 1024, 300)), jnp.float32)
    whole = pairwise._blocked(a, b, None)
    with pytest.MonkeyPatch.context() as patched:
        # A workspace of a block's sums of 64 columns: five runs of the columns.
        patched.setattr(pairwise, "workspace_bytes", lambda: 8 * 64 * 64 * 4)
        np.testing.assert_array_equal(np.asarray(pairwise._blocked(a, b, None)), np.asarray(whole))


@pytest.mark.parametrize("depth", [64, 128, 1024])
def test_a_dense_layers_float32_product_on_a_cpu_is_as_exact_as_a_vectorized_dot(depth):
    """A dense product in fixed float32 sums on a CPU (`precision.accumulated` of `jax.lax.dot_general`
    under `FP32_SUMS`, as row 4735's llama.cpp comparison runs Dew): bitwise `pairwise_dot_general`'s, and
    its value, its rows' gradient and its matrix's each on par (check 1) with a vectorized dot's
    accumulation on this CPU (`vectorized`: libggml's and a BLAS kernel's partial sums, a lane of each FMA
    in flight, summed as a tree), where XLA's one-chain dot was 2.0 to 6.1 times its error."""
    numbers = (((1,), (0,)), ((), ()))
    dense = functools.partial(precision.accumulated,
                              functools.partial(jax.lax.dot_general, dimension_numbers=numbers))

    def draw(seed):
        rng = np.random.default_rng((depth, seed))
        return tuple(jnp.asarray(rng.standard_normal(shape), jnp.float32)
                     for shape in ((256, depth), (depth, 96), (256, 96)))

    def ours(rows, matrix, cotangent):
        with jax.default_matmul_precision(precision.FP32_SUMS):
            value, pull = jax.vjp(dense, rows, matrix)
            return (value, *pull(cotangent))

    def reference(rows, matrix, cotangent):
        return (vectorized(rows, matrix), vectorized(cotangent, matrix.T), vectorized(rows.T, cotangent))

    def truth(rows, matrix, cotangent):
        return (rows @ matrix, cotangent @ matrix.T, rows.T @ cotangent)

    operands = draw(0)
    blocked = jax.vjp(functools.partial(pairwise.pairwise_dot_general, dimension_numbers=numbers),
                      *operands[:2])
    for mine, theirs in zip(jax.jit(ours)(*operands), (blocked[0], *blocked[1](operands[2])), strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))
    assert_as_exact(ours, reference, draw, f"a CPU dense product {depth} deep", seeds=8, truth_fn=truth)


def vectorized(a: jax.Array, b: jax.Array) -> jax.Array:
    """`a @ b` in float32 as a vectorized dot on this CPU sums it: `partials` sums (its `vector_lanes` times
    `FMA_LATENCY`), product `k` into sum `k mod partials`, each a chain in order, then a pairwise tree."""
    depth = a.shape[1]
    partials = min(depth, pairwise.vector_lanes() * pairwise.FMA_LATENCY)
    count = -(-depth // partials)
    pad = count * partials - depth
    strided_a = jnp.pad(a, ((0, 0), (0, pad))).reshape(a.shape[0], count, partials)
    strided_b = jnp.pad(b, ((0, pad), (0, 0))).reshape(count, partials, b.shape[1])
    sums = jax.lax.dot_general(strided_a, strided_b, (((1,), (0,)), ((2,), (1,))),
                               preferred_element_type=jnp.float32)
    while sums.shape[0] > 1:
        carried = sums[-1:] if sums.shape[0] % 2 else None
        even = sums[:sums.shape[0] - (carried is not None)]
        sums = even[0::2] + even[1::2]
        if carried is not None:
            sums = jnp.concatenate([sums, carried])
    return sums[0]
