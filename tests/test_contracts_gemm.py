"""Float-range and nested-derivative contracts of the products' seams: embedding lookups, bf16 and wide CPU
products, the grouped GEMM and its portable projection, the GEMM prologue and transposed hooks, a stored
expert primal and a router's report of its differentiation, the frozen FP8 product, the quantized
convolution, a broadcast summing its gradient wide (ledger 2641; the harness is tests/contract_support.py)."""

import contextlib
import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from arm_arithmetic import ahead_on_the_host
from contract_support import EDGES, FORWARD_ORDERS, assert_contract, assert_drawn_contract, normal_draw
from reference_error import Followed, Rounded, as_given, assert_as_exact, widened

from dew.nn import precision
from dew.nn.blocks import table_rows
from dew.nn.dense import Dense
from dew.nn.kernels import formats, gemm, generation, low_rank, projections
from dew.nn.kernels.templates.hooks import Hook, Side


@pytest.mark.parametrize("edge", EDGES)
def test_an_embedding_lookup_holds_every_order(edge):
    """`blocks.table_rows` (a gather whose reverse mode sums each token's cotangent into its row, row
    1869): every order runs in the table, the ids not differentiable, and each is finite wherever float64
    is a normal float32."""
    (table,) = normal_draw(1869, (11, 6), scale=EDGES[edge])
    ids = jnp.asarray([[0, 3, 3, 10], [7, 0, 2, 2]], jnp.int32)
    (direction,) = normal_draw(1870, table.shape, scale=EDGES[edge])
    (cotangent,) = normal_draw(1871, (2, 4, 6))
    assert_contract(lambda table: table_rows(table, ids, table.dtype), (table,), (direction,), cotangent,
                    f"an embedding lookup {edge}")


def test_an_embedding_lookup_over_ids_split_on_an_explicit_axis_takes_every_order():
    """`blocks.table_rows` over ids split along an Explicit data axis, where the rows are fewer than the
    table so the backward gathers them whole (row 3150): its gradient, the jvp of its gradient and the
    gradient of its gradient run and are finite, the ids' zero cotangent typed as the ids arrive, not as
    the gather holds them, under a loss of the rows alone and one weighing them by weights split as the
    ids are (row 3152)."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

    mesh = jax.make_mesh((4,), ("data",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:4])
    whole, split = NamedSharding(mesh, P()), NamedSharding(mesh, P("data", None))
    table, direction = (jax.device_put(jnp.asarray(drawn), whole)
                        for (drawn,) in (normal_draw(3150, (64, 8)), normal_draw(3151, (64, 8))))
    with jax.set_mesh(mesh):
        ids = jax.device_put(jnp.asarray(np.arange(16).reshape(4, 4) * 5 % 64, jnp.int32), split)

        weights = jax.device_put(jnp.asarray(normal_draw(3152, (4, 4, 8))[0]), NamedSharding(mesh, P("data")))
        losses = (lambda table: jnp.sum(table_rows(table, ids, table.dtype) ** 3),
                  # A batch-split loss's per-token terms, as a sharded training step differentiates them
                  # (row 3152): the lookup's backward's tangent typed as the table.
                  lambda table: jnp.sum(table_rows(table, ids, table.dtype) * weights))
        orders = []
        for loss in losses:
            gradient = jax.grad(loss)
            orders += [gradient(table), jax.jvp(gradient, (table,), (direction,))[1],
                       jax.grad(lambda table, gradient=gradient: jnp.vdot(gradient(table), direction))(table)]
    assert all(np.all(np.isfinite(np.asarray(order))) for order in orders)


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

    def exact(lhs, rhs):
        # The operands as the algorithm reads them, rounded to bf16, then multiplied in float64.
        return jnp.matmul(*(jnp.asarray(jnp.asarray(x, jnp.bfloat16), x.dtype) for x in (lhs, rhs)))

    def plain(lhs, rhs):
        # The operands rounded to bf16, their products summed in float32 by XLA's own dot.
        return jnp.matmul(*(jnp.asarray(jnp.asarray(x, jnp.bfloat16), x.dtype) for x in (lhs, rhs)),
                          preferred_element_type=jnp.float32)

    assert_contract(dot, primals, direction, cotangent, f"a bf16 dot {edge}", truth_fn=exact,
                    plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_a_wide_cpu_sum_of_products_holds_every_order(monkeypatch, edge):
    """`precision.accumulated` in a process running wide CPU arithmetic (`WIDE_CPU_ARITHMETIC`, its own
    custom derivative of the float64-summed product): every order runs, finite wherever float64 is a
    normal float32 and wherever XLA's plain product is, at values whose products near float32's ends."""
    monkeypatch.setenv(precision.WIDE_CPU_ARITHMETIC, "1")
    primals = normal_draw(343, (4, 16), (16, 5), scale=EDGES[edge])
    direction = normal_draw(344, (4, 16), (16, 5), scale=EDGES[edge])
    (cotangent,) = normal_draw(345, (4, 5))
    assert_contract(lambda lhs, rhs: precision.accumulated(jnp.matmul, lhs, rhs), primals, direction,
                    cotangent, f"a wide CPU sum of products {edge}", plain_fn=jnp.matmul)


@pytest.mark.parametrize("edge", EDGES)
def test_a_grouped_gemm_holds_every_order(edge):
    """The GEMM template's grouped product (`gemm.gemm`, its custom derivative through
    `generation.linear_tangent`): every order runs in the rows and the matrices, the group sizes not
    differentiable, finite wherever float64 is a normal float32 and wherever the plain grouped product is
    (each row's product with its group's matrix, by a mask, its float64 run the truth), at values whose
    products near float32's ends."""
    sizes = jnp.asarray([3, 0, 5], jnp.int32)
    groups = jnp.repeat(jnp.arange(3), sizes, total_repeat_length=8)
    def inputs(seed: int, scale: float) -> tuple:
        # Each stream salted apart (`normal_draw`'s salt): one seed's cotangent is no other's input.
        (cotangent,) = normal_draw([148, seed], (8, 4))
        return (normal_draw([146, seed], (8, 16), (3, 16, 4), scale=scale),
                normal_draw([147, seed], (8, 16), (3, 16, 4), scale=scale), cotangent)

    def grouped(rows, matrices):
        return gemm.gemm(rows, matrices, sizes, out_dtype=rows.dtype)

    def plain(rows, matrices):
        return jnp.einsum("rg,rk,gkn->rn", jax.nn.one_hot(groups, 3, dtype=rows.dtype), rows, matrices)

    assert_drawn_contract(grouped, inputs, EDGES[edge], f"a grouped GEMM {edge}", truth_fn=plain,
                          plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_matrices_side_by_side_hold_every_order(edge):
    """`gemm.gemm` of matrices side by side (`beside`: the rows' products with each, each its own output,
    its custom derivative the grouped GEMM's, through `generation.linear_tangent`): every order runs in the
    rows and the matrices, finite wherever float64 is a normal float32 and wherever the products apart are
    (each row's product with its group's matrix, by a mask, its float64 run the truth), at values whose
    products near float32's ends."""
    sizes = jnp.asarray([3, 0, 5], jnp.int32)
    groups = jnp.repeat(jnp.arange(3), sizes, total_repeat_length=8)
    shapes = ((8, 16), (3, 16, 4), (3, 16, 8))
    primals = normal_draw(6794, *shapes, scale=EDGES[edge])
    direction = normal_draw(6795, *shapes, scale=EDGES[edge])
    (cotangent,) = normal_draw(6796, (8, 12))

    def side_by_side(rows, first, second):
        return jnp.concatenate(gemm.gemm(rows, first, sizes, beside=(second,), out_dtype=rows.dtype), axis=-1)

    def plain(rows, first, second):
        hot = jax.nn.one_hot(groups, 3, dtype=rows.dtype)
        products = [jnp.einsum("rg,rk,gkn->rn", hot, rows, matrices) for matrices in (first, second)]
        return jnp.concatenate(products, axis=-1)

    assert_contract(side_by_side, primals, direction, cotangent, f"matrices side by side {edge}",
                    truth_fn=plain, plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_an_adapted_product_holds_every_reverse_order(edge):
    """`low_rank.low_rank` (LoRA's adapted product as one product over the depth its rank extends, its custom
    VJP declared through `derivatives.declared_derivative`): every reverse order, under a nested jit and a
    scan, in the rows, the kernel and both factors, finite wherever float64 is a normal float32 and wherever
    the plain adapted product (`x·W + s·(x·A)·B`, its float64 run the truth) is, at values whose products near
    float32's ends. It takes no forward order yet: a custom VJP has none (row 1304; the appended pair walked
    in place brings one)."""
    shapes = ((8, 16), (16, 12), (16, 4), (4, 12))
    primals = normal_draw(1304, *shapes, scale=EDGES[edge])
    direction = normal_draw(1305, *shapes, scale=EDGES[edge])
    (cotangent,) = normal_draw(1306, (8, 12))

    def adapted(rows, kernel, a, b):
        return low_rank.low_rank(rows, kernel, a, b, 0.5, out_dtype=rows.dtype)

    def plain(rows, kernel, a, b):
        return rows @ kernel + 0.5 * ((rows @ a) @ b)

    assert_contract(adapted, primals, direction, cotangent, f"an adapted product {edge}", truth_fn=plain,
                    plain_fn=plain, without=FORWARD_ORDERS)


@pytest.mark.parametrize("edge", EDGES)
def test_a_once_rounded_biased_layer_holds_every_order(edge):
    """A bfloat16 Dense whose bias is stored at fp32 (its product summed in fp32, the bias added and the sum
    rounded once: `dense._widened`'s declared tangent): every order, in the inputs, the kernel and the bias,
    finite wherever float64 is a normal float32 and wherever the plain formula (JAX's derivatives of the
    fp32 product plus the bias, rounded once) is, at values whose products near float32's ends."""
    layer = Dense(16, dtype=jnp.bfloat16, param_dtype=jnp.float32)
    primals = normal_draw(634, (8, 16), (16, 16), (16,), scale=EDGES[edge])
    direction = normal_draw(635, (8, 16), (16, 16), (16,), scale=EDGES[edge])
    (cotangent,) = normal_draw(636, (8, 16))

    def dense(x, kernel, bias):
        return layer.apply({"params": {"kernel": kernel, "bias": bias}}, x.astype(jnp.bfloat16))

    def plain(x, kernel, bias):
        product = jax.lax.dot(x.astype(jnp.bfloat16), kernel.astype(jnp.bfloat16),
                              preferred_element_type=jnp.float32)
        return (product + bias).astype(jnp.bfloat16)

    assert_contract(dense, primals, direction, cotangent, f"a once-rounded biased layer {edge}",
                    plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_a_biased_sum_holds_every_order(edge):
    """A product plus its bias as a biased layer adds them (`precision.biased`: the bias spread at fp32, its
    reverse mode through the rule's dual): every order, in the product and the bias, finite wherever float64
    is a normal float32 and wherever the plain sum is, at values near float32's ends."""
    primals = normal_draw(6471, (8, 16), (16,), scale=EDGES[edge])
    direction = normal_draw(6472, (8, 16), (16,), scale=EDGES[edge])
    (cotangent,) = normal_draw(6473, (8, 16))

    def plain(product, bias):
        return product + bias

    assert_contract(precision.biased, primals, direction, cotangent, f"a biased sum {edge}", plain_fn=plain)


def test_a_decoded_mxfp8_operand_preserves_its_linear_rule():
    codes = np.ones((2, 32), jnp.float8_e4m3fn)
    stored = np.asarray([[0], [127]], np.uint8).view(jnp.float8_e8m0fnu)
    incoming = np.broadcast_to(np.asarray([[2.0 ** 125], [0.5]], np.float32), codes.shape)

    @jax.jit
    def orders(values, scales, incoming):
        def decode(codes):
            return formats.MxFp8(codes, scales).decoded()

        value, tangent = jax.jvp(decode, (values,), (jnp.ones_like(values),))
        _, pull = jax.vjp(decode, values)
        return value, tangent, pull(incoming)[0]

    value, tangent, gradient = orders(codes, stored, incoming)
    expected = np.broadcast_to(np.asarray([[2.0 ** -127], [1]], np.float32), codes.shape)
    for found in (value, tangent):
        np.testing.assert_array_equal(np.asarray(found).view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(gradient, np.broadcast_to([[0.25], [0.5]], codes.shape))

    @jax.jit
    def exact_output(codes):
        def convert(values):
            return formats.MxFp8(values, stored).decoded(jnp.int32)

        return jax.jvp(convert, (codes,), (jnp.ones_like(codes),))

    integer, no_tangent = exact_output(codes)
    np.testing.assert_array_equal(integer, np.broadcast_to([[0], [1]], codes.shape))
    assert no_tangent.dtype == jax.dtypes.float0


def test_decoded_ternary_preserves_its_linear_scale_rule():
    values = np.tile(np.asarray([[-1, 1], [0, 0], [1, -1], [0, 0]], np.float32), (64, 1))
    weights = formats.TernaryWeights.of(values, scale=np.ones(2, np.float32))
    small = np.asarray([2.0 ** -127, -(2.0 ** -127)], np.float32)
    scale = np.ones_like(small)
    incoming = np.zeros_like(values)
    incoming[[0, 2]] = [[1, 2], [4, 8]]

    @jax.jit
    def orders(scale, direction, incoming):
        def decoded(scale):
            return formats.TernaryWeights(weights.words, scale, weights.form).decoded(jnp.float32)

        found, moved = jax.jvp(decoded, (scale,), (direction,))
        _, pull = jax.vjp(decoded, scale)
        return found, moved, pull(incoming)[0]

    found, tangent, gradient = orders(scale, small, incoming)
    np.testing.assert_array_equal(found, values)
    np.testing.assert_array_equal(np.asarray(tangent).view(np.uint32), (values * small).view(np.uint32))
    np.testing.assert_array_equal(gradient, [3, -6])

    @jax.jit
    def exact_output(s):
        def convert(scale):
            return formats.TernaryWeights(weights.words, scale, weights.form).decoded(jnp.int32)

        return jax.jvp(convert, (s,), (jnp.ones_like(s),))

    integer, no_tangent = exact_output(scale)
    np.testing.assert_array_equal(integer, values.astype(np.int32))
    assert no_tangent.dtype == jax.dtypes.float0


@pytest.mark.parametrize("edge", EDGES)
def test_an_mxfp8_projection_holds_every_reverse_order(edge):
    """`projections.mxfp8_projection` (the MXFP8 recipe, its custom VJP the recipe's own MX products, declared
    as a surrogate): every reverse order, under a nested jit and a scan, in the rows and the matrices, finite
    wherever float64 is a normal float32 and wherever the recipe's own fp32 sums are (tests/mx_matrix.py's
    `recipe_truth`, its float64 run the truth: a value quantized can round up past float32's largest where the
    unquantized one does not), at values whose products near float32's ends (row 2513). It takes no forward
    order: a custom VJP has none."""
    import mx_matrix

    group_blocks = jnp.asarray([1, 0, 1], jnp.int32)
    primals = normal_draw(2519, (256, 128), (3, 128, 128), scale=EDGES[edge])
    direction = normal_draw(2520, (256, 128), (3, 128, 128), scale=EDGES[edge])
    (cotangent,) = normal_draw(2521, (256, 128))

    def recipe(rows, matrices):
        return projections.mxfp8_projection(rows, matrices, group_blocks, out_dtype=rows.dtype)

    assert_contract(recipe, primals, direction, cotangent, f"an mxfp8 projection {edge}",
                    truth_fn=mx_matrix.recipe_truth(group_blocks),
                    plain_fn=mx_matrix.recipe_truth(group_blocks, jnp.float32), without=FORWARD_ORDERS)


@pytest.mark.parametrize("edge", EDGES)
def test_an_nvfp4_projection_holds_every_reverse_order(edge):
    """`projections.nvfp4_projection` (the NVFP4 recipe, its custom VJP the recipe's own NVFP4 products,
    declared as a surrogate): every reverse order, under a nested jit and a scan, in the rows and the
    matrices, finite wherever float64 is a normal float32 and wherever the recipe's own fp32 sums are
    (`mx_matrix.recipe_truth` of `formats.NvFp4Block`, its float64 run the truth), at values whose products
    near float32's ends (row 2513). It takes no forward order: a custom VJP has none."""
    import mx_matrix

    group_blocks = jnp.asarray([1, 0, 1], jnp.int32)
    primals = normal_draw(2522, (256, 128), (3, 128, 128), scale=EDGES[edge])
    direction = normal_draw(2523, (256, 128), (3, 128, 128), scale=EDGES[edge])
    (cotangent,) = normal_draw(2524, (256, 128))

    def recipe(rows, matrices):
        return projections.nvfp4_projection(rows, matrices, group_blocks, out_dtype=rows.dtype)

    truth = functools.partial(mx_matrix.recipe_truth, group_blocks, quantized=mx_matrix.nvfp4_values)
    assert_contract(recipe, primals, direction, cotangent, f"an nvfp4 projection {edge}", truth_fn=truth(),
                    plain_fn=truth(jnp.float32), without=FORWARD_ORDERS)


@pytest.mark.parametrize("edge", EDGES)
def test_a_blockwise_fp8_projection_holds_every_reverse_order(edge):
    """`projections.blockwise_fp8_projection` (DeepSeek-V3's blockwise FP8 recipe, its custom VJP the recipe's
    own FP8 products, declared through `derivatives.declared_derivative` as a surrogate): every reverse order,
    under a nested jit and a scan, in the rows and the matrices, finite wherever float64 is a normal float32
    and wherever the plain grouped product (its float64 run the truth) is, at values whose products near
    float32's ends (row 2513). It takes no forward order: a custom VJP has none."""
    group_blocks = jnp.asarray([1, 0, 1], jnp.int32)
    sizes = 128 * group_blocks
    primals = normal_draw(2513, (256, 128), (3, 128, 128), scale=EDGES[edge])
    direction = normal_draw(2514, (256, 128), (3, 128, 128), scale=EDGES[edge])
    (cotangent,) = normal_draw(2515, (256, 128))

    def recipe(rows, matrices):
        return projections.blockwise_fp8_projection(rows, matrices, group_blocks, out_dtype=rows.dtype)

    def plain(rows, matrices):
        return jax.lax.ragged_dot(rows, matrices, sizes)

    assert_contract(recipe, primals, direction, cotangent, f"a blockwise fp8 projection {edge}",
                    truth_fn=plain, plain_fn=plain, without=FORWARD_ORDERS)


def _swiglu(gate, up):
    return jax.nn.silu(gate) * up


@pytest.mark.parametrize("chain", ["blockwise fp8", "gated blockwise fp8", "gated delayed nvfp4"])
@pytest.mark.parametrize("edge", EDGES)
def test_a_recipe_chain_holds_every_reverse_order(edge, chain):
    """A recipe chain (an MLP's projections, the down projection's rows quantized in the up projection's
    epilogue; its custom VJP each projection's own surrogate): `projections.blockwise_chain` ungated through
    ReLU and gated through SwiGLU (the gate's product read by the up projection's epilogue; rows 8547,
    12689), and `projections.nvfp4_chain` gated at a delayed recipe's tensor scales, the draw's own largest
    magnitudes' as a history of this step holds them (row 12721): every reverse order, under a nested jit
    and a scan, in the rows and every matrix, finite wherever float64 is a normal float32 and wherever the
    plain product through the activation is, at values whose products near float32's ends. It takes no
    forward order: a custom VJP has none."""
    gated = chain.startswith("gated")
    shapes = ((256, 128), *((1, 128, 256),) * (1 + gated), (1, 256, 128))
    primals = normal_draw(8547, *shapes, scale=EDGES[edge])
    direction = normal_draw(8548, *shapes, scale=EDGES[edge])
    (cotangent,) = normal_draw(8549, (256, 128))
    activation = _swiglu if gated else jax.nn.relu

    def plain(rows, *matrices):
        *firsts, down = matrices
        return activation(*(rows @ matrix[0] for matrix in firsts)) @ down[0]

    rows, *firsts, _ = primals
    hidden = activation(*(rows @ matrix[0] for matrix in firsts))
    scales = tuple(projections.delayed_tensor_scale(projections.nvfp4_amax(each)[None])
                   for each in (rows, hidden))

    def recipe(rows, *matrices):
        *gate, up, down = matrices
        options = {"gate": gate[0] if gate else None, "out_dtype": rows.dtype}
        if chain.endswith("nvfp4"):
            return projections.nvfp4_chain(rows, up, down, activation, tensor_scales=scales, **options)
        return projections.blockwise_chain(rows, up, down, activation, **options)

    assert_contract(recipe, primals, direction, cotangent, f"a {chain} chain {edge}", truth_fn=plain,
                    plain_fn=plain, without=FORWARD_ORDERS)


@pytest.mark.parametrize("edge", EDGES)
def test_an_nvfp4_products_kernels_hold_every_order(edge, kernel="xla"):
    """`gemm.nvfp4_product` on its selector's `kernel` (`xla` here, the decoded matrix's product; `packed`,
    the template's mainloop decoding each stage in place through `generation.linear_tangent` as `gemm` is,
    on an H100 and an A100 in the GPU flow, tests/test_gpu_gemm.py: Mosaic GPU's interpreter runs its kernel
    through ordered callbacks, which no rematerialization differentiates): every order runs in the rows,
    finite wherever float64 is a normal float32 and wherever the plain product (the rows times the decoded
    values times the deferred factor, its float64 run the truth) is, at values whose products near float32's
    ends (row 3621). The frozen words and scales take no cotangent."""
    codes = np.random.default_rng(3621).integers(0, 16, size=(256, 64))
    e2m1 = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    magnitudes = np.where(codes >= 8, -1.0, 1.0) * e2m1[codes & 7]
    exponents = np.random.default_rng(3622).integers(-2, 3, size=(16, 64))
    # Packed and decoded on the host where a GPU tier part's prepare runs the cell, its device computing
    # nothing before the contract's sites (row 12377's class: the H100 and A100 flows' packed cells).
    with ahead_on_the_host():
        matrix = formats.NvFp4.of(magnitudes.astype(np.float32),
                                  np.exp2(exponents).astype(jnp.float8_e4m3fn), 0.5)
        values = np.asarray(matrix.decoded().astype(jnp.float32) * matrix.factor)
    primals = normal_draw(3623, (64, 256), scale=EDGES[edge])
    direction = normal_draw(3624, (64, 256), scale=EDGES[edge])
    (cotangent,) = normal_draw(3625, (64, 64))

    def product(rows):
        with generation.chosen(nvfp4=kernel):
            return gemm.nvfp4_product(rows, matrix, out_dtype=rows.dtype)

    def plain(rows):
        return rows @ values.astype(rows.dtype)

    assert_contract(product, primals, direction, cotangent, f"an NVFP4 product's {kernel} kernel {edge}",
                    truth_fn=plain, plain_fn=plain)


def test_an_nvfp4_products_cotangent_past_bf16s_largest_keeps_its_input_gradient_finite():
    """The NVFP4 product's packed kernel (its mainloop decoding each stage in place, interpreted as an H100's
    on a CPU) pulled back by a finite fp32 cotangent past bfloat16's largest, 3.4e38, through a matrix whose
    every value is 2^-9: its input gradient is the cotangent times the decoded values, finite as float32
    holds it (the cotangent taken as its saturating bf16 parts, `templates.product.formed`), where its own
    three-part split rounded the first part to infinity and its remainder to NaN."""
    # A tensor scale of 2^-8, so the factor the decode defers (`NvFp4.factor`, its owed 2^8 times the scale)
    # is one and the cotangent reaches the packed kernel's reverse as it is.
    matrix = formats.NvFp4.of(jnp.full((256, 64), 0.5, jnp.float32), jnp.ones((16, 64), jnp.float8_e4m3fn),
                           2.0 ** -8)
    rows = jnp.ones((8, 256), jnp.bfloat16)
    cotangent = jnp.zeros((8, 64), jnp.float32).at[0, 0].set(3.4e38)

    def product(rows):
        return gemm.nvfp4_product(rows, matrix, out_dtype=jnp.float32, interpret_on_cpu=True)

    with generation.assumed("sm90"), generation.chosen(nvfp4="packed"):
        (d_rows,) = jax.vjp(product, rows)[1](cotangent)
    # Every value 2^-9, each part's product exact, their sum the cotangent's times it, rounded once to the
    # rows' bfloat16.
    values = np.asarray(matrix.decoded(), np.float64) * float(matrix.factor)
    expected = jnp.asarray(np.asarray(cotangent, np.float64) @ values.T, jnp.float32).astype(rows.dtype)
    assert bool(jnp.all(jnp.isfinite(expected))) and float(jnp.max(jnp.abs(expected))) > 1e35
    np.testing.assert_array_equal(np.asarray(d_rows, np.float32), np.asarray(expected, np.float32))


GROUPS = jnp.asarray([3, 0, 5], jnp.int32)
"""A grouped product's sizes: eight rows, an empty middle expert."""


def masked_grouped_product(rows, matrices):
    """Each row's product with its group's matrix (`GROUPS`), by a mask: the grouped product as written."""
    groups = jnp.repeat(jnp.arange(3), GROUPS, total_repeat_length=8)
    return jnp.einsum("rg,rk,gkn->rn", jax.nn.one_hot(groups, 3, dtype=rows.dtype), rows, matrices)


@pytest.mark.parametrize("edge", [edge for edge in EDGES if edge != "near the largest"])
def test_a_portable_grouped_projection_holds_every_order(edge):
    """`xla_products.portable_projection` (its custom derivative, the rounded operands' products): every
    order finite wherever float64 is a normal float32 and the masked grouped product as written is. Near the
    largest its primal overflows a partial sum the masked product does not (ledger 2711), that row's head
    adds the edge."""
    from dew.nn.kernels import xla_products

    primals = normal_draw(1564, (8, 16), (3, 16, 4), scale=EDGES[edge])
    direction = normal_draw(1565, (8, 16), (3, 16, 4), scale=EDGES[edge])
    (cotangent,) = normal_draw(1566, (8, 4))
    def projected(rows, matrices):
        return xla_products.portable_projection(rows, matrices, GROUPS, None, None)

    assert_contract(projected, primals, direction, cotangent, f"a portable grouped projection {edge}",
                    truth_fn=masked_grouped_product, plain_fn=masked_grouped_product)


@jax.custom_vjp
def straight_through(scores):
    """A score hook through a custom VJP entry by entry: a rounding to bfloat16 whose cotangent passes
    unchanged."""
    return scores.astype(jnp.bfloat16).astype(scores.dtype)


straight_through.defvjp(lambda scores: (straight_through(scores), None), lambda _, cotangent: (cotangent,))


@pytest.mark.parametrize("edge", EDGES)
def test_a_transposed_hooks_custom_vjp_holds_every_reverse_order(edge):
    """A custom VJP within a hook a mainloop runs on transposed tiles (`layout.transposed`, its pullback held
    entry by entry, `_custom.held`): every reverse order, under a nested jit and a scan, finite wherever
    float64 is a normal float32 and the hook on the tile as its caller holds it is; it takes no forward
    order, as the call itself does not (ledger 2658)."""
    from dew.nn.kernels.templates import layout

    (tile,) = normal_draw(108, (6, 4), scale=EDGES[edge])
    (direction,) = normal_draw(109, (6, 4), scale=EDGES[edge])
    (cotangent,) = normal_draw(110, (6, 4))

    def plain(tile):
        return straight_through(tile.T).T

    assert_contract(layout.transposed(straight_through), (tile,), (direction,), cotangent,
                    f"a transposed hook's custom VJP {edge}", truth_fn=plain, plain_fn=plain,
                    without=FORWARD_ORDERS)


def rms_producer(lhs, scale):
    """An RMS norm as a GEMM's producer writes it: the operand over its root mean square, times its scale."""
    return lhs * jax.lax.rsqrt(jnp.mean(jnp.square(lhs), axis=-1, keepdims=True) + 1e-6) * scale


@pytest.mark.parametrize("edge", EDGES)
def test_a_gemm_prologue_holds_every_order(edge):
    """The GEMM template's prologue (`gemm._prologued`, a custom JVP whose tangent is the producer's own,
    as the fused GEMM hands it, `fused._whole_jvp`): every order finite wherever float64 is a normal float32
    and the producer as written is."""
    from dew.nn.kernels.gemm import _prologued

    prologued = _prologued(rms_producer, lambda *primals: lambda *tangents: jax.jvp(
        rms_producer, primals, tangents)[1])
    def inputs(seed: int, scale: float) -> tuple:
        # Each stream salted apart (`normal_draw`'s salt): one seed's cotangent is no other's input.
        (cotangent,) = normal_draw([396, seed], (6, 16))
        return (normal_draw([394, seed], (6, 16), (16,), scale=scale),
                normal_draw([395, seed], (6, 16), (16,), scale=scale), cotangent)

    assert_drawn_contract(prologued, inputs, EDGES[edge], f"a GEMM prologue {edge}", truth_fn=rms_producer,
                          plain_fn=rms_producer)


@pytest.mark.parametrize("edge", EDGES)
def test_a_fused_gemm_epilogue_holds_every_order(edge):
    """The fused GEMM's epilogue (`fused_gemm._declared`): its consumer's arithmetic with each call read
    through to its body, as the fuser tiles it, its derivative the consumer's as written, a gated product's
    SwiGLU (`activations.gated_hook`): every order finite wherever float64 is a normal float32 and the plain
    formula is, where the inlined logistic's own derivative is no number past exp's overflow."""
    from dew.nn import activations
    from dew.nn.kernels import fused_gemm

    hook = activations.gated_hook("swiglu")

    def consumer(values, acc):
        return hook(acc, values[0])

    shape = jax.ShapeDtypeStruct((6, 16), jnp.float32)
    treedef = jax.tree.structure((shape,))
    written, inlined = fused_gemm._rewritten_consumer(consumer, (shape,), shape)
    epilogue = fused_gemm._declared(
        lambda acc, *tiles: inlined(jax.tree.unflatten(treedef, list(tiles)), acc), written, treedef,
        jax.tree.structure(shape), shape)

    def plain(gate, up):
        return jax.nn.silu(gate) * up

    def inputs(seed: int, scale: float) -> tuple:
        (cotangent,) = normal_draw([399, seed], (6, 16))
        return (normal_draw([397, seed], (6, 16), (6, 16), scale=scale),
                normal_draw([398, seed], (6, 16), (6, 16), scale=scale), cotangent)

    assert_drawn_contract(epilogue, inputs, EDGES[edge], f"a fused GEMM epilogue {edge}", truth_fn=plain,
                          plain_fn=plain)


def softly_rectified(scores):
    """A score hook through a custom derivative entry by entry (jnp.logaddexp's)."""
    return jnp.logaddexp(scores, 0.5 * scores)


@pytest.mark.parametrize("edge", EDGES)
def test_a_transposed_hooks_custom_derivative_holds_every_order(edge):
    """A custom JVP within a hook a mainloop runs on transposed tiles (`layout.transposed`, its rule held
    entry by entry, `_custom`): every order, forward ones too, under a nested jit and a scan, finite wherever
    float64 is a normal float32 and the hook on the tile as its caller holds it is (ledger 2658: the held
    rule was a custom VJP whatever the call's own, so it took no forward order)."""
    from dew.nn.kernels.templates import layout

    (tile,) = normal_draw(105, (6, 4), scale=EDGES[edge])
    (direction,) = normal_draw(106, (6, 4), scale=EDGES[edge])
    (cotangent,) = normal_draw(107, (6, 4))

    def plain(tile):
        return softly_rectified(tile.T).T

    assert_contract(layout.transposed(softly_rectified), (tile,), (direction,), cotangent,
                    f"a transposed hook's custom derivative {edge}", truth_fn=plain, plain_fn=plain)


def test_a_transposed_hook_through_a_primitive_taking_an_entrywise_name_is_refused():
    """A hook through a primitive of its own named `exp`, an entrywise operation's name, that reads its
    operand whole is no operation `layout.transposed` takes entry for entry: refused, where it was bound on
    the transposed tile as JAX's own `exp` (train 587's review)."""
    from jax.extend.core import Primitive

    from dew.nn.kernels.templates import layout

    spread = Primitive("exp")
    spread.def_abstract_eval(lambda value: value)
    spread.def_impl(lambda value: jnp.broadcast_to(jnp.sum(value), value.shape))
    with pytest.raises(layout.Untransposable):
        layout.transposed(spread.bind)(jnp.ones((6, 4), jnp.float32))


@pytest.mark.parametrize("edge", [edge for edge in EDGES if edge != "near the largest"])
def test_a_stored_expert_primal_holds_every_order(edge):
    """An expert layer's primal over its parameters as stored (`moe._stored_primal`, differentiated over them
    widened): every order finite wherever float64 is a normal float32 and the run itself is. Near the
    largest its kernel gradient sums its products in an order that overflows where the run's own does not
    (ledger 2711), that row's head adds the edge."""
    from dew.nn import moe

    indices = jnp.zeros((6,), jnp.int32)

    def run(x, indices, parameters, weights):
        return (x @ parameters["kernel"]) * weights[:, None]

    def stored(x, kernel, weights):
        return moe._stored_primal(run, x, indices, {"kernel": kernel}, weights)

    def plain(x, kernel, weights):
        return run(x, indices, {"kernel": kernel}, weights)

    primals = normal_draw(597, (6, 16), (16, 4), (6,), scale=EDGES[edge])
    direction = normal_draw(598, (6, 16), (16, 4), (6,), scale=EDGES[edge])
    (cotangent,) = normal_draw(599, (6, 4))
    assert_contract(stored, primals, direction, cotangent, f"a stored expert primal {edge}", truth_fn=plain,
                    plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_a_routers_differentiation_report_holds_every_order(edge):
    """A fixed router's report of its differentiation (`moe._differentiated`, a custom JVP: its parameters
    themselves, their tangent passed through) over a router's kernel scaled to the edge: every order finite
    wherever float64 is a normal float32 and the plain identity is, and every order that differentiates
    the kernel reports it."""
    from dew.nn import moe

    reports = []
    (kernel,) = normal_draw(3211, (8, 4), scale=EDGES[edge])
    (direction,) = normal_draw(3212, (8, 4), scale=EDGES[edge])
    (cotangent,) = normal_draw(3213, (8, 4))

    def reported(kernel):
        return moe._differentiated(functools.partial(reports.append, edge), kernel)

    assert_contract(reported, (kernel,), (direction,), cotangent, f"a router's report {edge}",
                    plain_fn=lambda kernel: kernel)
    assert reports


LINEAR_EDGES = {"normal": 1.0, "near the largest": 2.0**122, "near the least normal": 2.0**-122}
"""The scales at which a product linear in one operand (the other a fixed unit draw) nears float32's ends: its
sums of sixteen products near the largest and the least normal, where `EDGES`' scale an operand by the square
root of each."""


@pytest.mark.parametrize("edge", LINEAR_EDGES)
def test_a_frozen_fp8_product_holds_every_order(edge):
    """A frozen FP8 kernel's product (`quantization.frozen_fp8_product`, its tangent the decoded kernel's):
    every order in the input finite wherever float64 is a normal float32 and the product with the kernel its
    codes decode to (each row's codes times its scale) is, the kernel a unit draw and the input at
    `LINEAR_EDGES`, so the product itself nears float32's ends: near the largest, the input's block scale
    applied before the kernel's had overflowed a product that did not (ledger 2715)."""
    from dew.quantization import FROZEN_FP8, frozen_fp8_leaves

    (kernel,) = normal_draw(475, (16, 8))
    leaves = frozen_fp8_leaves(kernel)
    # The forward's [out, in] E4M3 codes times each row's scale, decoded here apart from the product's own.
    decoded = (jnp.asarray(leaves["fp8_forward"]).astype(jnp.float32)
               * jnp.asarray(leaves["fp8_forward_scale"]).astype(jnp.float32)).T

    def frozen(x):
        from dew.quantization import frozen_fp8_product

        return frozen_fp8_product(x, *(leaves[name] for name in FROZEN_FP8))

    def plain(x):
        return x @ decoded.astype(x.dtype)

    (x,) = normal_draw(476, (5, 16), scale=LINEAR_EDGES[edge])
    (direction,) = normal_draw(477, (5, 16), scale=LINEAR_EDGES[edge])
    (cotangent,) = normal_draw(478, (5, 8))
    assert_contract(frozen, (x,), (direction,), cotangent, f"a frozen FP8 product {edge}", truth_fn=plain,
                    plain_fn=plain)


def int8_round_trip(values, kept: int, codes=None, met: "Int8Roundings | None" = None):
    """`values` quantized to int8 and back as Qwix's absmax calibration has it: one scale per index of axis
    `kept` (an activation's example, a kernel's output feature), the absolute maximum over the rest over
    127.5, 1 where that is below the root of its dtype's least normal (Qwix's guard), the codes rounded and
    clipped to int8's, or `codes` given (a followed truth's: an arm's decisions); the quotients rounded and
    the codes made, met (`met`) where given. The codes take no derivative and the scale its own, as Qwix
    computes them: neither holds the scale from differentiation."""
    over = tuple(axis for axis in range(values.ndim) if axis != kept)
    scale = jnp.max(jnp.abs(values), axis=over, keepdims=True) / 127.5
    scale = jnp.where(scale < jnp.sqrt(jnp.finfo(scale.dtype).tiny), 1, scale)
    if codes is None:
        quotients = values / scale
        codes = jnp.clip(jnp.round(quotients), -128, 127).astype(jnp.int8)
        if met is not None:
            met.met(quotients, codes)
    return jax.lax.stop_gradient(jnp.asarray(codes).astype(values.dtype)) * scale


class Int8Roundings:
    """The int8 roundings a run executed, each its quotients and codes as its program computed them, met on
    the host (`jax.debug.callback`) in execution order: an arm's rounding inputs and decisions at a site of
    followed roundings (`assert_as_exact`'s `followed`)."""

    def __init__(self):
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def met(self, quotients, codes) -> None:
        jax.debug.callback(lambda q, c: self.calls.append((np.asarray(q), np.asarray(c))), quotients, codes)

    def rounded(self, result, operands: dict) -> Rounded:
        """`result` beside each operand's rounding (`operands`, by name, its shape): the quotients it rounded
        a judged leaf (`rounded/<name>`), its codes a decision; an operand rounded more than once a run
        rounded alike each time, one decision of it."""
        judged, decided = {}, {}
        for name, shape in operands.items():
            calls = [(q, c) for q, c in self.calls if q.shape == tuple(shape)]
            for _, codes in calls[1:]:
                np.testing.assert_array_equal(codes, calls[0][1], err_msg=f"{name}'s {len(calls)} roundings")
            judged[f"rounded/{name}"], decided[name] = calls[0]
        self.calls.clear()
        return Rounded({"output": np.asarray(result), **judged}, decided)


def int8_rounds(judged: dict) -> dict:
    """A quantized convolution's int8 roundings of the quotients an arm rounded (its judged `rounded/<name>`
    leaves), Qwix's own (`numerics.convert_to`): round half to even, then int8's."""
    from qwix._src.core import numerics

    return {name.removeprefix("rounded/"): np.asarray(numerics.convert_to(jnp.asarray(value), jnp.int8))
            for name, value in judged.items() if name.startswith("rounded/")}


def convolved(x, kernel, groups: int):
    """A 3x3 NHWC convolution with an HWIO kernel in `groups` feature groups, padded to its input's size."""
    return jax.lax.conv_general_dilated(x, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"),
                                        feature_group_count=groups)


def quantized_convolution(x, kernel, groups: int, codes: tuple | None = None, met=None):
    """The convolution of `x` and `kernel` quantized to int8 and back (`int8_round_trip`: each example's
    scale, each output feature's), its tangent `conv(dx, Q(kernel)) + conv(Q(x), dkernel)`: the float
    convolution's at the dequantized operands, so its reverse mode is the transpose there, and a derivative of
    that goes through the operands' quantization, the scales' derivatives included, as the seam's backward
    reads the quantized operands its forward made. `codes` (the input's and the kernel's), where given, are
    the int8 codes it takes for its own (a followed truth's); `met` meets the roundings it makes."""
    given = (None, None) if codes is None else codes

    def held(x, kernel):
        return int8_round_trip(x, 0, given[0], met), int8_round_trip(kernel, 3, given[1], met)

    @jax.custom_jvp
    def convolution(x, kernel):
        return convolved(*held(x, kernel), groups)

    @convolution.defjvp
    def _(primals, tangents):
        (x, kernel), (along_x, along_kernel) = primals, tangents
        operands = held(x, kernel)
        return (convolved(*operands, groups),
                convolved(along_x, operands[1], groups) + convolved(operands[0], along_kernel, groups))

    return convolution(x, kernel)


def group_peaks(x, groups: int, features: int):
    """Each example's absolute maximum in each of `groups` feature groups of an NHWC input, 1 for an all-zero
    group, repeated over each group's share of `features` and held from differentiation: the scale a grouped
    quantized convolution divides its input by before it quantizes and multiplies its output by after."""
    grouped = x.reshape(*x.shape[:-1], groups, x.shape[-1] // groups)
    peak = jnp.max(jnp.abs(grouped), axis=(1, 2, 4))
    peak = jnp.where(peak > 0, peak, 1)
    return jax.lax.stop_gradient(jnp.repeat(peak, features // groups, axis=-1))[:, None, None, :]


@pytest.mark.parametrize(("groups", "edge"), list(itertools.product((1, 2), EDGES)))
def test_a_quantized_convolution_holds_every_order_it_takes(groups, edge):
    """A quantized convolution under Qwix's quantized training, one group and two (`quantization`'s
    `quantized_training`, a surrogate whose derivative is the float convolution's at the dequantized
    operands): every order finite wherever float64 is a normal float32 and the convolution of the operands
    quantized to int8 and back (each example's scale, each output feature's; a grouped input first over
    each example's peak in each group, restored on that group's outputs) is, its forward derivative the
    float convolution's at the dequantized operands, as the seam declares, its reverse mode that
    derivative's transpose, and a second derivative one of that through the quantization, the scales'
    included (`quantized_convolution`)."""
    from dew.nn.conv import Conv
    from dew.quantization import Quantization

    conv = Conv(features=8, kernel_size=(3, 3), padding="SAME", use_bias=False, feature_group_count=groups)
    quantized = Quantization().apply(conv)

    def convolved(x, kernel):
        return quantized.apply({"params": {"kernel": kernel}}, x)

    def plain(x, kernel):
        inward, outward = ((group_peaks(x, groups, width) for width in (x.shape[-1], kernel.shape[-1]))
                           if groups > 1 else (1, 1))
        return quantized_convolution(x / inward, kernel, groups) * outward

    primals = normal_draw(1134, (1, 6, 6, 4), (3, 3, 4 // groups, 8), scale=EDGES[edge])
    direction = normal_draw(1135, (1, 6, 6, 4), (3, 3, 4 // groups, 8), scale=EDGES[edge])
    (cotangent,) = normal_draw(1136, (1, 6, 6, 8))
    assert_contract(convolved, primals, direction, cotangent, f"a quantized convolution of {groups} {edge}",
                    truth_fn=plain, plain_fn=plain)


def test_a_quantized_convolutions_second_derivative_reads_its_scales(monkeypatch):
    """The quantized convolution's gradient in its kernel, differentiated in its input (a mixed second
    derivative): the gradient reads the input quantized, whose scale (each example's absolute maximum)
    moves with the input, so the derivative goes through it, as the independent oracle's does
    (`quantized_convolution`): on par with the oracle (check 1), each against the float64 oracle that
    rounds where it rounded (`followed`: two programs' quotients a rounding apart near a code's midpoint
    choose its two codes, which no truth that rounds for itself holds both to; row 6939). Dew's roundings
    are Qwix's own (`numerics.convert_to`), each met as its program computed it."""
    from qwix._src.core import numerics

    from dew.nn.conv import Conv
    from dew.quantization import Quantization

    quantized = Quantization().apply(Conv(features=8, kernel_size=(3, 3), padding="SAME", use_bias=False))
    roundings = {"dew": Int8Roundings(), "oracle": Int8Roundings()}
    qwix_rounding = numerics.convert_to

    def met_rounding(x, qtype, noise_fn=None):
        codes = qwix_rounding(x, qtype, noise_fn)
        if codes.dtype == jnp.int8:
            roundings["dew"].met(x, codes)
        return codes

    monkeypatch.setattr(numerics, "convert_to", met_rounding)

    def mixed(convolution):
        def run(x, kernel, along_x, cotangent):
            def kernel_gradient(x):
                return jax.grad(lambda k: jnp.sum(convolution(x, k) * cotangent))(kernel)
            return jax.jvp(kernel_gradient, (x,), (along_x,))[1]
        return run

    def dew_convolution(x, kernel):
        return quantized.apply({"params": {"kernel": kernel}}, x)

    def arm(name, convolution):
        program = jax.jit(mixed(convolution))

        @as_given
        def run(x, kernel, along_x, cotangent):
            jax.effects_barrier()  # every earlier run's roundings met before this run's are
            roundings[name].calls.clear()
            result = jax.block_until_ready(program(x, kernel, along_x, cotangent))
            jax.effects_barrier()
            return roundings[name].rounded(result, {"x": x.shape, "kernel": kernel.shape})
        return run

    @as_given
    def truth(decisions, x, kernel, along_x, cotangent):
        with jax.enable_x64(new_val=True):
            held = (decisions["x"], decisions["kernel"])
            wide = widened((x, kernel, along_x, cotangent))
            scaled = {name: values / jnp.where(scale < jnp.sqrt(jnp.finfo(scale.dtype).tiny), 1, scale)
                      for name, values, scale in
                      (("x", wide[0], jnp.max(jnp.abs(wide[0]), axis=(1, 2, 3), keepdims=True) / 127.5),
                       ("kernel", wide[1], jnp.max(jnp.abs(wide[1]), axis=(0, 1, 2), keepdims=True) / 127.5))}
            output = mixed(lambda x, k: quantized_convolution(x, k, 1, held))(*wide)
            return {"output": np.asarray(output), **{f"rounded/{name}": np.asarray(value)
                                                      for name, value in scaled.items()}}

    def draw(seed):
        return tuple(normal_draw(2000 + seed, (1, 6, 6, 4), (3, 3, 4, 8), (1, 6, 6, 4), (1, 6, 6, 8)))

    oracle = arm("oracle", lambda x, k: quantized_convolution(x, k, 1, met=roundings["oracle"]))
    assert_as_exact(arm("dew", dew_convolution), oracle, draw,
                    "a quantized convolution's mixed second derivative", seeds=8,
                    followed=Followed(truth, int8_rounds))


@pytest.mark.parametrize("edge", EDGES)
def test_a_broadcast_summing_its_gradient_wide_holds_every_order(edge):
    """`precision.broadcast_summed` (a broadcast whose gradient sums in float32 and rounds once to its
    value's dtype, as torch's autograd reduces a bf16 gradient, which tier 3's broadcasts run): every
    order runs, forward mode's included (ledger 2653), finite wherever float64 is a normal float32, at
    values near float32's ends, beside the plain broadcast."""
    value, direction = normal_draw(959, (3, 1, 4), (3, 1, 4), scale=EDGES[edge])
    (cotangent,) = normal_draw(961, (3, 5, 4))

    def broadcast(value):
        return precision.broadcast_summed(value, (3, 5, 4), (0, 1, 2))

    def plain(value):
        return jnp.broadcast_to(value, (3, 5, 4))

    assert_contract(broadcast, (value,), (direction,), cotangent, f"a wide-summed broadcast {edge}",
                    truth_fn=plain, plain_fn=plain)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_a_broadcast_sums_its_gradient_wide_at_every_order(dtype):
    """`precision.broadcast_summed`'s transpose sums in float32 and rounds once wherever a derivative
    forms one: a weighted square's gradient, the gradient's tangent (forward over reverse) and its gradient
    (reverse over reverse) each sum 4096 terms of the value's narrow dtype, every product exact, to the
    float32 sum rounded once, where a sum in the narrow dtype rounds its partial sums (ledger 2653)."""
    rows = 4096
    weights = jnp.asarray(1 + (np.arange(rows * 4).reshape(rows, 4) % 7) * 2.0**-7, dtype)
    ones = jnp.ones(4, dtype)

    def loss(value):
        return jnp.sum(weights * jnp.square(precision.broadcast_summed(value, (rows, 4), (1,))))

    # Each term 2 w, exact in the narrow dtype, summed exactly in float32 and rounded once.
    expected = np.asarray(2 * np.asarray(weights, np.float32).sum(0)).astype(dtype)
    gradient = jax.grad(loss)(ones)
    tangent = jax.jvp(jax.grad(loss), (ones,), (ones,))[1]
    curvature = jax.grad(lambda value: jnp.vdot(jax.grad(loss)(value), ones).astype(jnp.float32))(ones)
    for derivative in (gradient, tangent, curvature):
        np.testing.assert_array_equal(np.asarray(derivative), expected, strict=True)
    narrow = np.asarray(2 * weights).sum(0, dtype=dtype)
    assert not np.array_equal(narrow, expected), "a narrow sum rounds these terms apart"


@pytest.mark.parametrize("edge", EDGES)
def test_a_portable_formed_rule_holds_every_order(edge):
    """An XLA program whose last operand is a parameter through the rule (`derivatives.portable_formed`:
    its value and every derivative XLA's of it, its reverse mode's cotangents leaving together): every order,
    forward and reverse, under a nested jit and a scan, in the rows and the parameter, finite wherever
    float64 is a normal float32 and wherever the program itself (its float64 run the truth) is, at values
    near float32's ends."""
    from dew.nn.derivatives import portable_formed

    primals = normal_draw(6471, (4, 32, 64), (64,), scale=EDGES[edge])
    direction = normal_draw(6472, (4, 32, 64), (64,), scale=EDGES[edge])
    (cotangent,) = normal_draw(6473, (4, 32, 64))

    def plain(rows, weight):
        return jnp.tanh(rows) * weight + rows * rows

    assert_contract(portable_formed(plain), primals, direction, cotangent, f"a portable formed rule {edge}",
                    plain_fn=plain)


# A projection's forms: a dense one's rows against a matrix, a batched one's (DeepSeek V4's grouped
# `o_a_proj`) each batch member's rows against its own matrix.
XLA_FORMS = {"dense": ((4, 32, 64), (64, 48), (4, 32, 48), (((2,), (0,)), ((), ()))),
             "batched": ((4, 32, 64), (4, 64, 48), (4, 32, 48), (((2,), (1,)), ((0,), (0,))))}


@pytest.mark.parametrize("form", XLA_FORMS)
@pytest.mark.parametrize("edge", EDGES)
def test_an_xla_projection_in_its_layer_holds_every_order(edge, form):
    """A projection's product no launch takes (`projections._in_layer_product`: XLA's own dot, its reverse
    mode's cotangents leaving together, `derivatives.with_gradients_formed`), dense or batched: every order,
    forward and reverse, under a nested jit and a scan, in the rows and the matrix, finite wherever float64
    is a normal float32 and wherever the plain dot (its float64 run the truth) is, at values whose products
    near float32's ends."""
    rows, matrix, out, dimensions = XLA_FORMS[form]
    primals = normal_draw(6027, rows, matrix, scale=EDGES[edge])
    direction = normal_draw(6028, rows, matrix, scale=EDGES[edge])
    (cotangent,) = normal_draw(6029, out)
    taken = projections._in_layer_product(jax.lax.dot_general, dimensions, None, None)

    def product(rows, matrix):
        return taken(rows, (matrix,))

    def plain(rows, matrix):
        return jax.lax.dot_general(rows, matrix, dimensions)

    assert_contract(product, primals, direction, cotangent, f"an xla projection in its layer {edge}",
                    plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_a_biased_xla_projection_in_its_layer_holds_every_order(edge):
    """A biased projection's product no launch takes, its bias added inside its rule
    (`projections._in_layer_product` with a bias: the input's cotangent leaving with the kernel's and the
    bias's gradients, one barrier for both): every order, forward and reverse, under a nested jit and a
    scan, in the rows, the matrix and the bias, finite wherever float64 is a normal float32 and wherever the
    plain dot plus its bias (its float64 run the truth) is, at values whose products near float32's ends."""
    rows, matrix, out, dimensions = XLA_FORMS["dense"]
    primals = normal_draw(6740, rows, matrix, out[-1:], scale=EDGES[edge])
    direction = normal_draw(6741, rows, matrix, out[-1:], scale=EDGES[edge])
    (cotangent,) = normal_draw(6742, out)
    taken = projections._in_layer_product(jax.lax.dot_general, dimensions, None, None)

    def product(rows, matrix, bias):
        return taken(rows, (matrix, bias))

    def plain(rows, matrix, bias):
        return jax.lax.dot_general(rows, matrix, dimensions) + bias

    assert_contract(product, primals, direction, cotangent, f"a biased xla projection in its layer {edge}",
                    plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_a_symmetric_gram_and_polynomial_hold_every_reverse_order(edge):
    """`gemm.symmetric_gram` (`X @ X^T` and `X^T @ X`) and `gemm.symmetric_polynomial`
    (`c0 * I + c1 * A + c2 * (A @ A)`), their custom VJPs declared through
    `derivatives.declared_derivative`: every reverse order, under a nested jit and a scan, finite wherever
    float64 is a normal float32 and wherever the plain formula is (row 11941)."""
    (x_fat,) = normal_draw(11941, (2, 6, 10), scale=EDGES[edge])
    (d_fat,) = normal_draw(11942, (2, 6, 10), scale=EDGES[edge])
    (c_gram,) = normal_draw(11943, (2, 6, 6))

    def plain_gram(operand):
        return jax.lax.dot_general(operand, operand, (((2,), (2,)), ((0,), (0,))))

    assert_contract(lambda operand: gemm.symmetric_gram(operand, out_dtype=operand.dtype),
                    (x_fat,), (d_fat,), c_gram, f"symmetric_gram fat {edge}",
                    truth_fn=plain_gram, plain_fn=plain_gram, without=FORWARD_ORDERS)

    (x_tall,) = normal_draw(11944, (2, 10, 6), scale=EDGES[edge])
    (d_tall,) = normal_draw(11945, (2, 10, 6), scale=EDGES[edge])

    def plain_gram_t(operand):
        return jax.lax.dot_general(operand, operand, (((1,), (1,)), ((0,), (0,))))

    assert_contract(lambda operand: gemm.symmetric_gram(operand, transpose=True, out_dtype=operand.dtype),
                    (x_tall,), (d_tall,), c_gram, f"symmetric_gram tall {edge}",
                    truth_fn=plain_gram_t, plain_fn=plain_gram_t, without=FORWARD_ORDERS)

    mat, coeffs = normal_draw(11946, (2, 6, 6), (3,), scale=EDGES[edge])
    d_mat, d_coeffs = normal_draw(11947, (2, 6, 6), (3,), scale=EDGES[edge])

    def plain_poly(operand, poly):
        sq = jax.lax.dot_general(operand, operand, (((2,), (1,)), ((0,), (0,))))
        return poly[0] * jnp.eye(operand.shape[-1], dtype=operand.dtype) + poly[1] * operand + poly[2] * sq

    assert_contract(lambda operand, poly: gemm.symmetric_polynomial(operand, poly, out_dtype=operand.dtype),
                    (mat, coeffs), (d_mat, d_coeffs), c_gram, f"symmetric_polynomial {edge}",
                    truth_fn=plain_poly, plain_fn=plain_poly, without=FORWARD_ORDERS)

    # Large-cotangent non-overflow witness: dy + dy.T would overflow 2**127 to inf before multiplying 0.25.
    small_x = jnp.asarray([[[0.25]]], jnp.float32)
    big_dy = jnp.asarray([[[2.0 ** 127]]], jnp.float32)
    for transpose in (False, True):
        _, pull = jax.vjp(lambda operand, t=transpose: gemm.symmetric_gram(operand, transpose=t), small_x)
        (dx_big,) = pull(big_dy)
        np.testing.assert_array_equal(np.asarray(dx_big), np.asarray([[[2.0 ** 126]]], jnp.float32))

    # Native Mosaic SYRK VJP contract (row 12036): bf16 under assumed sm90 + chosen mosaic + interpret_on_cpu,
    # directly exercising `_symmetric_gram_bwd` and `_symmetric_poly_bwd`.
    (x_bf,) = normal_draw(11948, (1, 64, 64), scale=EDGES[edge])
    (d_bf,) = normal_draw(11949, (1, 64, 64), scale=EDGES[edge])
    (c_bf,) = normal_draw(11950, (1, 64, 64))
    sym_bf = 0.5 * (x_bf + jnp.swapaxes(x_bf, -1, -2))
    d_sym_bf = 0.5 * (d_bf + jnp.swapaxes(d_bf, -1, -2))
    (_, p_bf) = normal_draw(11951, (1,), (3,), scale=EDGES[edge])
    (_, dp_bf) = normal_draw(11952, (1,), (3,), scale=EDGES[edge])

    import contextlib

    from dew.nn.sharding import automatic_axes

    def native_gram_bf(operand):
        choose = contextlib.nullcontext() if automatic_axes() else generation.chosen(syrk="mosaic")
        with generation.assumed("sm90"), choose:
            return gemm.symmetric_gram(operand.astype(jnp.bfloat16), out_dtype=jnp.bfloat16,
                                       interpret_on_cpu=True)

    def plain_gram_bf(operand):
        x = operand.astype(jnp.bfloat16)
        return jax.lax.dot_general(x, x, (((2,), (2,)), ((0,), (0,))),
                                   preferred_element_type=jnp.float32).astype(jnp.bfloat16)

    assert_contract(native_gram_bf, (x_bf,), (d_bf,), c_bf, f"symmetric_gram native mosaic bf16 {edge}",
                    truth_fn=plain_gram, plain_fn=plain_gram_bf, without=FORWARD_ORDERS)

    def native_poly_bf(operand, poly):
        choose = contextlib.nullcontext() if automatic_axes() else generation.chosen(syrk="mosaic")
        with generation.assumed("sm90"), choose:
            return gemm.symmetric_polynomial(operand.astype(jnp.bfloat16), poly, out_dtype=jnp.bfloat16,
                                             interpret_on_cpu=True)

    def plain_poly_bf(operand, poly):
        x = operand.astype(jnp.bfloat16)
        sq = jax.lax.dot_general(x, x, (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32)
        p = poly.astype(jnp.float32)
        return (p[0] * jnp.eye(64, dtype=jnp.float32) + p[1] * x.astype(jnp.float32) + p[2] * sq).astype(
            jnp.bfloat16)

    assert_contract(native_poly_bf, (sym_bf, p_bf), (d_sym_bf, dp_bf), c_bf,
                    f"symmetric_polynomial native mosaic bf16 {edge}",
                    truth_fn=plain_poly, plain_fn=plain_poly_bf, without=FORWARD_ORDERS)



@pytest.mark.parametrize("edge", EDGES)
def test_a_compute_prologued_gemm_holds_every_order(edge):
    """`gemm.gemm` with a named `compute` beside `prologue` and `applied` over an fp32 master weight
    (row 1958): every order, under a nested jit and a scan, finite wherever float64 is a normal float32 and
    wherever plain `portable_projection(prologue(gate, up), down)` is."""
    from dew.nn import activations
    from dew.nn.kernels import xla_products

    hook = activations.gated_hook("swiglu", None)
    group_sizes = jnp.asarray([5, 0, 3], jnp.int32)
    gate, up, down = normal_draw(1958, (8, 6), (8, 6), (3, 6, 4), scale=EDGES[edge])
    d_gate, d_up, d_down = normal_draw(1959, (8, 6), (8, 6), (3, 6, 4), scale=EDGES[edge])
    (cotangent,) = normal_draw(1960, (8, 4))

    def fused(g, u, w):
        hidden = hook(g.astype(jnp.bfloat16), u.astype(jnp.bfloat16)).astype(jnp.bfloat16)
        return gemm.gemm(g.astype(jnp.bfloat16), w, group_sizes,
                         prologue=Hook(hook, (Side(u.astype(jnp.bfloat16), ("rows", "depth")),)),
                         applied=hidden, compute=jnp.bfloat16, out_dtype=jnp.bfloat16)

    def plain(g, u, w):
        act = hook(g.astype(jnp.bfloat16), u.astype(jnp.bfloat16)).astype(jnp.bfloat16)
        return xla_products.portable_projection(act, w, group_sizes, jnp.bfloat16, None, jnp.bfloat16)

    def truth(g, u, w):
        act = hook(g, u)
        return jax.lax.ragged_dot(act, w, group_sizes)

    assert_contract(fused, (gate, up, down), (d_gate, d_up, d_down), cotangent,
                    f"compute prologued gemm {edge}", truth_fn=truth, plain_fn=plain)


def test_an_fp8_codes_product_holds_every_order():
    """`gemm.fp8_codes_product` (FP8 codes' product through the template's converting launch on an A100,
    `linear_tangent`, row 8548): every order in its rows, each a draw cast to E4M3, against the codes'
    exact values' plain product; the frozen matrix takes no cotangent."""
    (raw,) = normal_draw(8548, (128, 256), scale=1.0)
    matrix = raw.astype(jnp.float8_e4m3fn)
    (x,) = normal_draw(8552, (64, 256), scale=1.0)
    (dx,) = normal_draw(8553, (64, 256), scale=1.0)
    (cotangent,) = normal_draw(8554, (64, 128))

    from dew.nn.sharding import automatic_axes

    def product(rows):
        choose = contextlib.nullcontext() if automatic_axes() else generation.chosen(grouped_matmul="mosaic")
        with choose:
            return gemm.fp8_codes_product(rows.astype(jnp.float8_e4m3fn), matrix, out_dtype=jnp.float32,
                                          interpret_on_cpu=True)

    def plain(rows):
        return jax.lax.dot_general(rows.astype(jnp.float8_e4m3fn).astype(jnp.float32),
                                   matrix.astype(jnp.float32), (((1,), (1,)), ((), ())),
                                   precision=jax.lax.Precision.HIGHEST)

    without = () if generation.placed_platform() == "gpu" else ("jvp of grad, remat", "second jvp, remat")
    with generation.assumed("sm80"):
        assert_contract(product, (x,), (dx,), cotangent, "fp8 codes product", truth_fn=plain, plain_fn=plain,
                        without=without)


@pytest.mark.parametrize("edge", EDGES)
def test_a_ternary_weight_product_holds_every_order(edge):
    """`gemm.ternary_weight_product` (2-bit packed BitNet b1.58 weights decoded inline, `linear_tangent`):
    every order in `lhs`, plus interpreted packed & portable routes with `quantize_activations=True` checked
    against independent integer-dot forward and STE activation + scale tangent oracles (row 1464)."""
    import dataclasses

    from dew.nn import precision as prec
    from dew.nn.kernels import formats

    _check_ternary_weight_product_contract(edge)
    if edge != "normal":
        return
    raw_w = normal_draw(1464, (256, 64), scale=1.0)[0]
    weights = formats.TernaryWeights.of(raw_w, scale=jnp.float32(0.75))
    (x,) = normal_draw(1465, (64, 256), scale=1.0)
    (dx,) = normal_draw(1466, (64, 256), scale=1.0)
    # Independent integer-dot forward oracle and unquantized STE activation + scale tangent oracles.
    w_bf16 = weights.unscaled(jnp.bfloat16)
    x_f32 = x.astype(jnp.float32)
    peak = jnp.max(jnp.abs(x_f32), axis=-1)
    safe_peak = jnp.where(peak > 0, peak, jnp.float32(1.0))
    x_int = jnp.clip(jnp.round(prec.divided(x_f32, safe_peak[:, None]) * jnp.float32(127.0)),
                     -128.0, 127.0).astype(jnp.bfloat16)
    dot_int = jax.lax.dot(x_int, w_bf16, preferred_element_type=jnp.float32)
    prod_int = prec.divided(dot_int, jnp.float32(127.0))
    fwd_oracle = prec.twice_scaled(prod_int, peak[:, None], weights.scale)
    x_tan_oracle = jax.lax.dot(dx.astype(jnp.bfloat16), w_bf16,
                               preferred_element_type=jnp.float32) * weights.scale
    s_tan_oracle = prec.twice_scaled(prod_int, peak[:, None], jnp.float32(1.0))
    for kernel in ("mosaic", "xla"):
        with (generation.assumed("sm90"), generation.chosen(grouped_matmul=kernel),
              generation.routes() as seen):
            def fn(a, s):
                return gemm.ternary_weight_product(
                    a, dataclasses.replace(weights, scale=s), quantize_activations=True,
                    out_dtype=jnp.float32, interpret_on_cpu=True)

            out, dy_x = jax.jvp(lambda a: fn(a, weights.scale), (x,), (dx,))
            _, dy_s = jax.jvp(lambda s: fn(x, s), (weights.scale,), (jnp.float32(1.0),))
        assert any(r.kernel == kernel for r in seen), (kernel, seen)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(fwd_oracle))
        np.testing.assert_array_equal(np.asarray(dy_x), np.asarray(x_tan_oracle))
        np.testing.assert_array_equal(np.asarray(dy_s), np.asarray(s_tan_oracle))


def _check_ternary_weight_product_contract(edge: str) -> None:
    from dew.nn.kernels import formats

    depth = 256
    raw_w = normal_draw(1464, (depth, 64), scale=1.0)[0]
    weights = formats.TernaryWeights.of(raw_w, scale=jnp.float32(0.5))
    decoded = weights.decoded(jnp.bfloat16)
    (x,) = normal_draw(8549, (64, depth), scale=EDGES[edge])
    (dx,) = normal_draw(8550, (64, depth), scale=EDGES[edge])
    (cotangent,) = normal_draw(8551, (64, 64))

    from dew.nn.sharding import automatic_axes

    def product(rows):
        choose = contextlib.nullcontext() if automatic_axes() else generation.chosen(grouped_matmul="mosaic")
        with choose:
            return gemm.ternary_weight_product(rows, weights, quantize_activations=False,
                                               out_dtype=jnp.float32, interpret_on_cpu=True)

    def plain(rows):
        return jax.lax.dot(rows.astype(jnp.bfloat16), decoded, preferred_element_type=jnp.float32)

    without = () if generation.placed_platform() == "gpu" else ("jvp of grad, remat", "second jvp, remat")
    with generation.assumed("sm90"):
        assert_contract(product, (x,), (dx,), cotangent, f"ternary weight product {edge}",
                        truth_fn=plain, plain_fn=plain, without=without)


def test_a_scaled_partial_keeps_the_plain_products_overflow_and_non_finite_values():
    """`precision.scaled_partial` picks the order of its three products, never the value: a product whose true
    value overflows still overflows (one rounding past the overflow midpoint whose scales' product first
    reads fp32's largest among them), a NaN, an infinity and a zero operand give the plain product's value,
    and its derivative in each operand is the plain product's where every order is normal (rows 13274,
    13291)."""
    from dew.nn.precision import scaled_partial

    f32 = functools.partial(jnp.asarray, dtype=jnp.float32)
    for product, first, second in ((2.0 ** 100, 2.0 ** 20, 2.0 ** 10),
                                   (21 * 2.0 ** 27, 8388626 * 2.0 ** 25, 12782613 * 2.0 ** 25)):
        assert float(scaled_partial(f32(product), f32(first), f32(second))) == np.inf
    for product, first, second in ((np.nan, 1.0, 1.0), (np.inf, 2.0, 0.5), (0.0, 2.0 ** 100, 2.0 ** 100),
                                   (np.inf, 0.0, 1.0), (1.0, np.inf, 0.5)):
        got = float(scaled_partial(f32(product), f32(first), f32(second)))
        want = float(np.float32(product) * np.float32(first) * np.float32(second))
        assert got == want or (np.isnan(got) and np.isnan(want)), (product, first, second, got, want)
    grads = jax.grad(lambda *ops: scaled_partial(*ops), argnums=(0, 1, 2))(f32(3.0), f32(0.5), f32(4.0))
    assert tuple(float(g) for g in grads) == (2.0, 12.0, 1.5)


def test_a_scaled_partial_joins_its_scales_where_neither_order_holds_and_differentiates_that_order():
    """`precision.scaled_partial` in float16 (whose subnormals a CPU keeps, where it flushes fp32's): a
    subnormal row scale 2^-16 beside a column scale 2^15 of a 2.25 partial, each order's intermediate
    subnormal or past the largest, is 1.125, the scales' product first (row 13274), and its derivatives are
    that order's alone, a cotangent of 1/4 giving 1/8, 18432 and 9 2^-20, where a cotangent through the
    unchosen overflowing order reads NaN (row 13291's review)."""
    from dew.nn.precision import scaled_partial

    f16 = functools.partial(jnp.asarray, dtype=jnp.float16)
    operands = (f16(2.25), f16(2.0 ** -16), f16(2.0 ** 15))
    value, pull = jax.vjp(scaled_partial, *operands)
    assert float(value) == 1.125
    assert tuple(float(g) for g in pull(f16(0.25))) == (0.125, 18432.0, 9 * 2.0 ** -20)
