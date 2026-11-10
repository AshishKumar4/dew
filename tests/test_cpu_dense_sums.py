"""A CPU's float32 dense products summed no worse than a vectorized dot (`dew.nn.pairwise`, row 4863): the
block each contraction takes from the CPU's facts, the sum each product takes by the bytes it moves (blocked
and pairwise, or in float64), the products a dense layer forms taking it in fixed float32 sums, and their
error against float64, forward and both gradients, on par with a vectorized dot's."""

import functools
import itertools

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


def test_a_rows_product_alone_is_on_par_with_its_product_among_others():
    """`_summed`'s product of a row alone (a decode step's) and of a call's first 100 rows, each on par (check
    1) with the same rows' products in one call of 256 (a prefill's), and the matrix's gradient over those
    rows likewise, at 1024 deep: two programs, whose bits XLA does not share (testing.md, row 3177), each
    blocked and pairwise."""
    def draw(seed):
        rng = np.random.default_rng((4863, seed))
        return (jnp.asarray(rng.standard_normal((1, 256, 1024)), jnp.float32),
                jnp.asarray(rng.standard_normal((1, 1024, 512)), jnp.float32))

    def apart(a, b):
        return (*(pairwise._summed(a[:, row:row + 1], b, None) for row in (0, 7, 255)),
                pairwise._summed(a[:, :100], b, None))

    def together(a, b):
        whole = pairwise._summed(a, b, None)
        return (*(whole[:, row:row + 1] for row in (0, 7, 255)), whole[:, :100])

    def truth(a, b):
        whole = a @ b
        return (*(whole[:, row:row + 1] for row in (0, 7, 255)), whole[:, :100])

    assert_as_exact(jax.jit(apart), jax.jit(together), draw, "a row's CPU product alone and among others",
                    seeds=8, truth_fn=truth)


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


def _integer_gradients(product, *operands):
    """`product`'s value and its operands' gradients under a cotangent of small integers drawn from the
    value (each sum exact where the operands are small integers too)."""
    value, pull = jax.vjp(product, *operands)
    return value, *pull(jnp.round(4 * jnp.cos(value)))


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize(("rows_shape", "matrix_shape"), [((12, 4), (4, 3)), ((4, 3, 4), (4, 2, 3))],
                         ids=["rows", "batched-rows-headed-columns"])
def test_a_pairwise_product_on_an_explicit_data_split_sums_each_devices_share(rows_shape, matrix_shape):
    """A projection's pairwise product (`pairwise_dot_general`) on a mesh whose Explicit data axis splits its
    rows over two devices, the rows a matrix or a batch of sequences, the columns flat or a head's (a
    `DenseGeneral`'s, whose result a reshape splits off the split rows, row 5422): its value and both
    gradients, the matrix's contracting the split rows a device's share at a time (`_summed`), each the
    dot's own where every sum is exact (small integers) and placed as XLA's product and gradients place
    them, each share summed in two-row blocks."""
    rng = np.random.default_rng(4863)
    rows, matrix = (jnp.asarray(rng.integers(-3, 4, size=shape), jnp.float32)
                    for shape in (rows_shape, matrix_shape))
    numbers = (((len(rows_shape) - 1,), (0,)), ((), ()))
    product = functools.partial(pairwise.pairwise_dot_general, dimension_numbers=numbers)

    mesh = jax.make_mesh((2,), ("data",), (jax.sharding.AxisType.Explicit,), devices=jax.devices()[:2])
    with pytest.MonkeyPatch.context() as patched, jax.set_mesh(mesh):
        # Two-row blocks: a device's six rows its matrix's gradient sums as three blocks.
        patched.setattr(pairwise, "contraction_block", lambda depth, lanes=None: min(depth, 2))
        split = jax.device_put(rows, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data")))
        summed = jax.jit(functools.partial(_integer_gradients, product))(split, matrix)
        plain = jax.jit(functools.partial(_integer_gradients, functools.partial(
            jax.lax.dot_general, dimension_numbers=numbers)))(split, matrix)
    for got, want in zip(summed, plain, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        assert jax.typeof(got).sharding.spec == jax.typeof(want).sharding.spec


@pytest.mark.mesh(devices=2)
def test_a_headed_projection_on_an_explicit_data_split_trains_in_a_users_process(monkeypatch):
    """A `DenseGeneral` projection into heads (an attention block's, `projected_heads`) of a batch of
    sequences an Explicit data axis splits, in a process as a user's runs, without the suite's wide CPU
    arithmetic (`WIDE_CPU_ARITHMETIC`), so a CPU sums it pairwise (`pairwise_dot_general`; row 5422: a
    reshape of its result splitting the split rows raised ShardingTypeError): its value and the inputs' and
    kernel's gradients those of the plain contraction, every sum exact (small integers), each placed as
    that one's."""
    from dew.nn.dense import DenseGeneral

    monkeypatch.delenv(precision.WIDE_CPU_ARITHMETIC, raising=False)
    rng = np.random.default_rng(5422)
    inputs = jnp.asarray(rng.integers(-3, 4, size=(4, 3, 4)), jnp.float32)
    layer = DenseGeneral((2, 3), use_bias=False)
    kernel = jnp.asarray(rng.integers(-3, 4, size=(4, 2, 3)), jnp.float32)

    def projected(kernel, inputs):
        return layer.apply({"params": {"kernel": kernel}}, inputs)

    def plain(kernel, inputs):
        return jnp.einsum("btd,dhk->bthk", inputs, kernel)

    mesh = jax.make_mesh((2,), ("data",), (jax.sharding.AxisType.Explicit,), devices=jax.devices()[:2])
    with jax.set_mesh(mesh):
        split = jax.device_put(inputs, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data")))
        summed = jax.jit(functools.partial(_integer_gradients, projected))(kernel, split)
        want = jax.jit(functools.partial(_integer_gradients, plain))(kernel, split)
    for got, wanted in zip(summed, want, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(wanted))
        assert jax.typeof(got).sharding.spec == jax.typeof(wanted).sharding.spec


@pytest.mark.parametrize("depth", [96, 1024])
def test_a_grouped_gemms_float32_products_on_a_cpu_are_on_par_with_each_groups_dense_product(depth):
    """The grouped GEMM's portable products on a CPU in fixed float32 sums (`gemm.gemm` and `gemm.gemm_t`,
    their ragged dots through `precision.accumulated`, row 4863's GEMM remainder): each group's rows' product
    and its input gradient, and the kernel gradient of one group of every row, each on par (check 1) with
    `pairwise_dot_general`'s of that group's rows alone, over groups of rows no block divides, one empty and
    one of a single row: two programs, whose bits XLA does not share (testing.md, row 3177; they differed by
    1 to 4 ulp on an x86-64-v3 host and agreed on a v4 one)."""
    sizes = [70, 0, 1, 129, 37]
    starts = np.cumsum([0, *sizes])
    numbers = (((1,), (0,)), ((), ()))
    group_sizes, whole = jnp.asarray(sizes, jnp.int32), jnp.asarray([sum(sizes)], jnp.int32)

    def draw(seed):
        rng = np.random.default_rng((depth, seed))
        return tuple(jnp.asarray(rng.standard_normal(shape), jnp.float32)
                     for shape in ((sum(sizes), depth), (len(sizes), depth, 40), (sum(sizes), 40)))

    def grouped(x, w, dy):
        with jax.default_matmul_precision(precision.FP32_SUMS):
            return (gemm.gemm(x, w, group_sizes, out_dtype=jnp.float32),
                    gemm.gemm(dy, w, group_sizes, transpose_rhs=True, out_dtype=jnp.float32),
                    gemm.gemm_t(x, dy, whole, out_dtype=jnp.float32)[0])

    def summed(x, w, dy, product):
        out = [product(x[start:end], w[g]) for g, (start, end) in enumerate(itertools.pairwise(starts))]
        dx = [product(dy[start:end], jnp.swapaxes(w[g], 0, 1))
              for g, (start, end) in enumerate(itertools.pairwise(starts))]
        return jnp.concatenate(out), jnp.concatenate(dx), product(x.T, dy)

    dense = functools.partial(summed, product=functools.partial(pairwise.pairwise_dot_general,
                                                                dimension_numbers=numbers))
    assert_as_exact(jax.jit(grouped), jax.jit(dense), draw, f"a CPU grouped product {depth} deep", seeds=8,
                    truth_fn=functools.partial(summed, product=lambda a, b: a @ b))


def test_a_grouped_float32_product_of_no_rows_is_the_empty_product():
    """A grouped float32 projection of no rows over two groups on a CPU in fixed float32 sums (`gemm.gemm`,
    `gemm.gemm_t`, their pairwise sums): its output and input gradient no rows, and its kernel gradient, each
    group's contraction over no rows, zero for every group, the empty product XLA's own ragged dot gives."""
    tokens, cotangent = jnp.zeros((0, 40), jnp.float32), jnp.zeros((0, 16), jnp.float32)
    kernel = jnp.asarray(np.random.default_rng(5194).integers(-3, 4, size=(2, 40, 16)), jnp.float32)
    sizes = jnp.zeros((2,), jnp.int32)
    with jax.default_matmul_precision(precision.FP32_SUMS):
        out, pull = jax.vjp(lambda t, k: gemm.gemm(t, k, sizes, out_dtype=jnp.float32), tokens, kernel)
        pulled_tokens, pulled_kernel = pull(cotangent)
        gradient = jax.jit(lambda t, c: gemm.gemm_t(t, c, sizes, out_dtype=jnp.float32))(tokens, cotangent)
    for got, want in ((out, cotangent), (pulled_tokens, tokens), (pulled_kernel, jnp.zeros_like(kernel)),
                      (gradient, jnp.zeros_like(kernel))):
        assert got.dtype == want.dtype
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.mesh(devices=2)
def test_a_grouped_float32_projection_on_an_explicit_data_split_traces_and_places_as_xlas():
    """A float32 grouped projection, its input gradient and its kernel gradient (`gemm.gemm`, `gemm.gemm_t`,
    their CPU branch `pairwise.pairwise_ragged_dot_general`, which a platform choice traces on every platform)
    of rows an Explicit data axis splits over two devices: each traced and run (JAX infers no placement for a
    ragged dot, so each block's states its own, `grouped_matmul.ragged_placement`, and a kernel gradient's
    gathers read whole rows), each value XLA's ragged products' bit for bit over small integers, and placed as
    `ragged_placement` places them."""
    from dew.nn.kernels.grouped_matmul import ragged_placement

    rng = np.random.default_rng(4864)
    tokens, cotangent = (jnp.asarray(rng.integers(-3, 4, size=shape), jnp.float32)
                         for shape in ((64, 40), (64, 16)))
    kernel = jnp.asarray(rng.integers(-3, 4, size=(2, 40, 16)), jnp.float32)
    sizes = jnp.asarray([40, 24], jnp.int32)
    rows = jax.lax.RaggedDotDimensionNumbers((((1,), (1,)), ((), ())), (0,), (0,))
    groups = jax.lax.RaggedDotDimensionNumbers((((0,), (0,)), ((), ())), (0,), ())

    def summed(tokens, cotangent):
        out, pull = jax.vjp(lambda t, k: gemm.gemm(t, k, sizes, out_dtype=jnp.float32), tokens, kernel)
        return out, *pull(cotangent), gemm.gemm_t(tokens, cotangent, sizes, out_dtype=jnp.float32)

    def plain(tokens, cotangent):
        def ragged(lhs, rhs, numbers, sources):
            return jax.lax.ragged_dot_general(lhs, rhs, sizes, numbers, preferred_element_type=jnp.float32,
                                              out_sharding=ragged_placement(sources))

        transposed = jnp.swapaxes(kernel, 1, 2)
        gradient = ragged(tokens, cotangent, groups, (None, (tokens, 1), (cotangent, 1)))
        return (ragged(tokens, kernel, rows, ((tokens, 0), (kernel, 2))),
                ragged(cotangent, transposed, rows, ((cotangent, 0), (transposed, 2))), gradient, gradient)

    mesh = jax.make_mesh((2,), ("data",), (jax.sharding.AxisType.Explicit,), devices=jax.devices()[:2])
    with pytest.MonkeyPatch.context() as patched, jax.set_mesh(mesh):
        patched.setattr(pairwise, "contraction_block", lambda depth, lanes=None: min(depth, 8))
        split = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
        placed = (jax.device_put(tokens, split), jax.device_put(cotangent, split))
        with jax.default_matmul_precision(precision.FP32_SUMS):
            got_all = jax.jit(summed)(*placed)
        want_all = jax.jit(plain)(*placed)
    for got, want in zip(got_all, want_all, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        assert jax.typeof(got).sharding.spec == jax.typeof(want).sharding.spec
