"""A CPU's float32 dense products summed no worse than a vectorized dot (`dew.nn.pairwise`, row 4863): the
block each contraction takes from the CPU's facts, the sum each product takes by the bytes it moves (blocked
and pairwise, or in float64), the products a dense layer forms taking it in fixed float32 sums, and their
error against float64, forward and both gradients, on par with a vectorized dot's."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew.nn import pairwise, precision
from dew.nn.kernels import gemm


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


def test_a_rows_product_is_its_own_whatever_rows_share_its_call():
    """`_summed` sums every row alike, blocked and pairwise (`_blocked`), so a row's product is the same bits
    alone as among 255 others, as a decode step's row is its prefill's and a packed document's its own
    row's; and the matrix's gradient over a call's rows the same bits split into two calls' shares as
    whole, at 1024 deep."""
    rng = np.random.default_rng(4863)
    a = jnp.asarray(rng.standard_normal((1, 256, 1024)), jnp.float32)
    b = jnp.asarray(rng.standard_normal((1, 1024, 512)), jnp.float32)
    whole = np.asarray(pairwise._summed(a, b, None))
    for row in (0, 7, 255):
        np.testing.assert_array_equal(np.asarray(pairwise._summed(a[:, row:row + 1], b, None)),
                                      whole[:, row:row + 1])
    np.testing.assert_array_equal(np.asarray(pairwise._summed(a[:, :100], b, None)), whole[:, :100])


def test_a_layers_projection_and_every_accumulated_contraction_sum_as_a_vectorized_dot():
    """A layer's projection (`gemm.dot_general`, a Dense's product past the template) and every float32
    contraction `accumulated` sums (`float32_dot_general`, a product the algebra derives, a mixer's
    `float32_einsum`) take the CPU's vectorized-dot sum: one order for sums that cancel each other."""
    numbers = (((1,), (0,)), ((), ()))
    operands = (jnp.ones((4, 96), jnp.float32), jnp.ones((96, 8), jnp.float32))

    def chooses(product) -> bool:
        def walked(jaxpr):
            for equation in jaxpr.eqns:
                yield equation.primitive.name
                for value in equation.params.values():
                    inner = getattr(value, "jaxpr", value)
                    if isinstance(inner, jax.extend.core.Jaxpr):
                        yield from walked(inner)

        traced = jax.make_jaxpr(product)(*operands)
        return "platform_index" in set(walked(traced.jaxpr))

    with jax.default_matmul_precision(precision.FP32_SUMS):
        assert chooses(lambda a, b: gemm.dot_general(precision.float32_dot_general)(a, b, numbers))
        assert chooses(lambda a, b: precision.float32_dot_general(a, b, numbers))
        assert chooses(lambda a, b: precision.float32_einsum("rd,dc->rc", a, b))


def test_a_mixers_einsum_on_a_cpu_is_the_pairwise_product():
    """A mixer's contraction in fixed float32 sums (`float32_einsum` under `FP32_SUMS`, a Mamba layer's chunk
    einsum `bcdn,bcn->bcd`) on a CPU is `pairwise_dot_general`'s product over the einsum's axes, bitwise, its
    value and both operands' gradients: one sum for a model's mixers and its projections, where XLA's one
    FMA chain left Jamba's and Falcon-H1's logits and gradients apart from their sources'."""
    rng = np.random.default_rng(0)
    states = jnp.asarray(rng.standard_normal((2, 3, 96, 64)), jnp.float32)
    readout = jnp.asarray(rng.standard_normal((2, 3, 64)), jnp.float32)
    cotangent = jnp.asarray(rng.standard_normal((2, 3, 96)), jnp.float32)

    def ours(states, readout):
        with jax.default_matmul_precision(precision.FP32_SUMS):
            return precision.float32_einsum("bcdn,bcn->bcd", states, readout)

    def theirs(states, readout):
        return pairwise.pairwise_dot_general(states, readout, (((3,), (2,)), ((0, 1), (0, 1))))

    def pulled(function):
        value, pull = jax.vjp(jax.jit(function), states, readout)
        return (value, *pull(cotangent))

    for mine, expected in zip(pulled(ours), pulled(theirs), strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(expected))


@pytest.mark.parametrize("depth", [64, 128, 1024])
def test_a_dense_layers_float32_product_on_a_cpu_is_as_exact_as_a_vectorized_dot(depth):
    """A dense layer's product in fixed float32 sums on a CPU (`gemm.dot_general` of `float32_dot_general`
    under `FP32_SUMS`, as row 4735's llama.cpp comparison runs Dew): bitwise `pairwise_dot_general`'s, and
    its value, its rows' gradient and its matrix's each on par (check 1) with a vectorized dot's
    accumulation on this CPU (`vectorized`: libggml's and a BLAS kernel's partial sums, a lane of each FMA
    in flight, summed as a tree), where XLA's one-chain dot was 2.0 to 6.1 times its error."""
    numbers = (((1,), (0,)), ((), ()))

    def dense(rows, matrix):
        # A Dense layer's product past the template (`gemm.dot_general`).
        return gemm.dot_general(precision.float32_dot_general)(rows, matrix, numbers)

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


@pytest.mark.mesh(devices=2)
def test_a_pairwise_product_on_an_explicit_data_split_sums_each_devices_share():
    """A projection's pairwise product (`pairwise_dot_general`) on a mesh whose Explicit data axis splits its
    rows over two devices: its value and both gradients, the matrix's contracting the split rows a device's
    share at a time (`_summed`), each the dot's own where every sum is exact (small integers) and placed as
    XLA's product and gradients place them, each share summed in two-row blocks."""
    rng = np.random.default_rng(4863)
    rows, matrix = (jnp.asarray(rng.integers(-3, 4, size=shape), jnp.float32)
                    for shape in ((12, 4), (4, 3)))
    product = functools.partial(pairwise.pairwise_dot_general, dimension_numbers=(((1,), (0,)), ((), ())))

    def gradients(product, rows, matrix):
        value, pull = jax.vjp(product, rows, matrix)
        return value, *pull(jnp.round(4 * jnp.cos(value)))

    mesh = jax.make_mesh((2,), ("data",), (jax.sharding.AxisType.Explicit,), devices=jax.devices()[:2])
    with pytest.MonkeyPatch.context() as patched, jax.set_mesh(mesh):
        # Two-row blocks: a device's six rows its matrix's gradient sums as three blocks.
        patched.setattr(pairwise, "contraction_block", lambda depth, lanes=None: min(depth, 2))
        split = jax.device_put(rows, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data")))
        summed = jax.jit(functools.partial(gradients, product))(split, matrix)
        plain = jax.jit(functools.partial(gradients, jnp.matmul))(split, matrix)
    for got, want in zip(summed, plain, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        assert jax.typeof(got).sharding.spec == jax.typeof(want).sharding.spec


def test_a_grouped_gemms_float32_products_on_a_cpu_sum_each_groups_as_its_dense_product():
    """The grouped GEMM's portable products on a CPU in fixed float32 sums (`gemm.gemm` and `gemm.gemm_t`, their
    ragged dots through `precision.accumulated`, row 4863's GEMM remainder): each group's rows' product and its
    input gradient's bitwise `pairwise_dot_general`'s of that group's rows alone, the depth blocked and summed
    as a tree as a dense product's is, over groups of rows no block divides, one empty and one of a single row,
    at 96 and 1024 deep; and the kernel gradient of one group of every row bitwise the dense one's, its rows
    blocked from the group's first and the blocks' sums added as `_tree` adds them."""
    sizes = [70, 0, 1, 129, 37]
    starts = np.cumsum([0, *sizes])
    rng = np.random.default_rng(4863)
    numbers = (((1,), (0,)), ((), ()))
    for depth in (96, 1024):
        x = jnp.asarray(rng.standard_normal((sum(sizes), depth)), jnp.float32)
        w = jnp.asarray(rng.standard_normal((len(sizes), depth, 40)), jnp.float32)
        dy = jnp.asarray(rng.standard_normal((sum(sizes), 40)), jnp.float32)
        group_sizes = jnp.asarray(sizes, jnp.int32)
        whole = jnp.asarray([sum(sizes)], jnp.int32)
        with jax.default_matmul_precision(precision.FP32_SUMS):
            out = jax.jit(lambda x, w: gemm.gemm(x, w, group_sizes, out_dtype=jnp.float32))(x, w)
            dx = jax.jit(lambda dy, w: gemm.gemm(dy, w, group_sizes, transpose_rhs=True,
                                                 out_dtype=jnp.float32))(dy, jnp.swapaxes(w, 1, 2))
            dw = jax.jit(lambda x, dy: gemm.gemm_t(x, dy, whole, out_dtype=jnp.float32))(x, dy)
        for g, (start, end) in enumerate(zip(starts[:-1], starts[1:], strict=True)):
            rows, cotangent = x[start:end], dy[start:end]
            np.testing.assert_array_equal(np.asarray(out[start:end]),
                                          np.asarray(pairwise.pairwise_dot_general(rows, w[g], numbers)))
            np.testing.assert_array_equal(np.asarray(dx[start:end]), np.asarray(
                pairwise.pairwise_dot_general(cotangent, jnp.swapaxes(w[g], 0, 1), numbers)))
        np.testing.assert_array_equal(np.asarray(dw[0]),
                                      np.asarray(pairwise.pairwise_dot_general(x.T, dy, numbers)))
