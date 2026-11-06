"""Float-range and nested-derivative contracts of the products' seams: embedding lookups, bf16 and wide CPU
products, the grouped GEMM and its portable projection, the GEMM prologue and transposed hooks, a stored
expert primal, the frozen FP8 product, the quantized convolution, a broadcast summing its gradient wide
(ledger 2641; the harness is tests/contract_support.py)."""

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from contract_support import EDGES, FORWARD_ORDERS, assert_contract, normal_draw
from reference_error import assert_as_exact

from dew.nn import precision
from dew.nn.blocks import table_rows
from dew.nn.dense import Dense
from dew.nn.kernels import gemm, generation


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
    primals = normal_draw(146, (8, 16), (3, 16, 4), scale=EDGES[edge])
    direction = normal_draw(147, (8, 16), (3, 16, 4), scale=EDGES[edge])
    (cotangent,) = normal_draw(148, (8, 4))

    def grouped(rows, matrices):
        return gemm.gemm(rows, matrices, sizes, out_dtype=rows.dtype)

    def plain(rows, matrices):
        return jnp.einsum("rg,rk,gkn->rn", jax.nn.one_hot(groups, 3, dtype=rows.dtype), rows, matrices)

    assert_contract(grouped, primals, direction, cotangent, f"a grouped GEMM {edge}", truth_fn=plain,
                    plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_an_adapted_product_holds_every_reverse_order(edge):
    """`gemm.low_rank` (LoRA's adapted product as one product over the depth its rank extends, its custom VJP
    declared through `derivatives.declared_derivative`): every reverse order, under a nested jit and a scan,
    in the rows, the kernel and both factors, finite wherever float64 is a normal float32 and wherever the
    plain adapted product (`x·W + s·(x·A)·B`, its float64 run the truth) is, at values whose products near
    float32's ends. It takes no forward order yet: a custom VJP has none (row 1304; the appended pair walked
    in place brings one)."""
    shapes = ((8, 16), (16, 12), (16, 4), (4, 12))
    primals = normal_draw(1304, *shapes, scale=EDGES[edge])
    direction = normal_draw(1305, *shapes, scale=EDGES[edge])
    (cotangent,) = normal_draw(1306, (8, 12))

    def adapted(rows, kernel, a, b):
        return gemm.low_rank(rows, kernel, a, b, 0.5, out_dtype=rows.dtype)

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
def test_an_mxfp8_projection_holds_every_reverse_order(edge):
    """`gemm.mxfp8_projection` (the MXFP8 recipe, its custom VJP the recipe's own MX products, declared as a
    surrogate): every reverse order, under a nested jit and a scan, in the rows and the matrices, finite
    wherever float64 is a normal float32 and wherever the recipe's own fp32 sums are (tests/mx_matrix.py's
    `recipe_truth`, its float64 run the truth: a value quantized can round up past float32's largest where
    the unquantized one does not), at values whose products near float32's ends (row 2513). It takes no
    forward order: a custom VJP has none."""
    import mx_matrix

    group_blocks = jnp.asarray([1, 0, 1], jnp.int32)
    primals = normal_draw(2519, (256, 128), (3, 128, 128), scale=EDGES[edge])
    direction = normal_draw(2520, (256, 128), (3, 128, 128), scale=EDGES[edge])
    (cotangent,) = normal_draw(2521, (256, 128))

    def recipe(rows, matrices):
        return gemm.mxfp8_projection(rows, matrices, group_blocks, out_dtype=rows.dtype)

    assert_contract(recipe, primals, direction, cotangent, f"an mxfp8 projection {edge}",
                    truth_fn=mx_matrix.recipe_truth(group_blocks),
                    plain_fn=mx_matrix.recipe_truth(group_blocks, jnp.float32), without=FORWARD_ORDERS)


@pytest.mark.parametrize("edge", EDGES)
def test_a_blockwise_fp8_projection_holds_every_reverse_order(edge):
    """`gemm.blockwise_fp8_projection` (DeepSeek-V3's blockwise FP8 recipe, its custom VJP the recipe's own
    FP8 products, declared through `derivatives.declared_derivative` as a surrogate): every reverse order,
    under a nested jit and a scan, in the rows and the matrices, finite wherever float64 is a normal float32
    and wherever the plain grouped product (its float64 run the truth) is, at values whose products near
    float32's ends (row 2513). It takes no forward order: a custom VJP has none."""
    group_blocks = jnp.asarray([1, 0, 1], jnp.int32)
    sizes = 128 * group_blocks
    primals = normal_draw(2513, (256, 128), (3, 128, 128), scale=EDGES[edge])
    direction = normal_draw(2514, (256, 128), (3, 128, 128), scale=EDGES[edge])
    (cotangent,) = normal_draw(2515, (256, 128))

    def recipe(rows, matrices):
        return gemm.blockwise_fp8_projection(rows, matrices, group_blocks, out_dtype=rows.dtype)

    def plain(rows, matrices):
        return jax.lax.ragged_dot(rows, matrices, sizes)

    assert_contract(recipe, primals, direction, cotangent, f"a blockwise fp8 projection {edge}",
                    truth_fn=plain, plain_fn=plain, without=FORWARD_ORDERS)


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
    block = jnp.asarray(np.exp2(exponents), jnp.float8_e4m3fn)
    matrix = gemm.NvFp4.of(jnp.asarray(magnitudes, jnp.float32), block, 0.5)
    values = matrix.decoded().astype(jnp.float32) * matrix.factor
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


GROUPS = jnp.asarray([3, 0, 5], jnp.int32)
"""A grouped product's sizes: eight rows, an empty middle expert."""


def masked_grouped_product(rows, matrices):
    """Each row's product with its group's matrix (`GROUPS`), by a mask: the grouped product as written."""
    groups = jnp.repeat(jnp.arange(3), GROUPS, total_repeat_length=8)
    return jnp.einsum("rg,rk,gkn->rn", jax.nn.one_hot(groups, 3, dtype=rows.dtype), rows, matrices)


@pytest.mark.parametrize("edge", [edge for edge in EDGES if edge != "near the largest"])
def test_a_portable_grouped_projection_holds_every_order(edge):
    """`grouped_matmul.portable_projection` (its custom derivative, the rounded operands' products): every
    order finite wherever float64 is a normal float32 and the masked grouped product as written is. Near the
    largest its primal overflows a partial sum the masked product does not (ledger 2711), that row's head
    adds the edge."""
    from dew.nn.kernels import grouped_matmul

    primals = normal_draw(1564, (8, 16), (3, 16, 4), scale=EDGES[edge])
    direction = normal_draw(1565, (8, 16), (3, 16, 4), scale=EDGES[edge])
    (cotangent,) = normal_draw(1566, (8, 4))
    def projected(rows, matrices):
        return grouped_matmul.portable_projection(rows, matrices, GROUPS, None, None)

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
    primals = normal_draw(394, (6, 16), (16,), scale=EDGES[edge])
    direction = normal_draw(395, (6, 16), (16,), scale=EDGES[edge])
    (cotangent,) = normal_draw(396, (6, 16))
    assert_contract(prologued, primals, direction, cotangent, f"a GEMM prologue {edge}",
                    truth_fn=rms_producer, plain_fn=rms_producer)


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


def int8_round_trip(values, kept: int):
    """`values` quantized to int8 and back as Qwix's absmax calibration has it: one scale per index of axis
    `kept` (an activation's example, a kernel's output feature), the absolute maximum over the rest over
    127.5, 1 where that is below the root of its dtype's least normal (Qwix's guard), the codes rounded and
    clipped to int8's. The codes take no derivative and the scale its own, as Qwix computes them: neither
    holds the scale from differentiation."""
    over = tuple(axis for axis in range(values.ndim) if axis != kept)
    scale = jnp.max(jnp.abs(values), axis=over, keepdims=True) / 127.5
    scale = jnp.where(scale < jnp.sqrt(jnp.finfo(scale.dtype).tiny), 1, scale)
    return jax.lax.stop_gradient(jnp.clip(jnp.round(values / scale), -128, 127)) * scale


def convolved(x, kernel, groups: int):
    """A 3x3 NHWC convolution with an HWIO kernel in `groups` feature groups, padded to its input's size."""
    return jax.lax.conv_general_dilated(x, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"),
                                        feature_group_count=groups)


@functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
def quantized_convolution(x, kernel, groups: int):
    """The convolution of `x` and `kernel` quantized to int8 and back (`int8_round_trip`: each example's
    scale, each output feature's), its tangent `conv(dx, Q(kernel)) + conv(Q(x), dkernel)`: the float
    convolution's at the dequantized operands, so its reverse mode is the transpose there, and a derivative of
    that goes through the operands' quantization, the scales' derivatives included, as the seam's backward
    reads the quantized operands its forward made."""
    return convolved(int8_round_trip(x, 0), int8_round_trip(kernel, 3), groups)


@quantized_convolution.defjvp
def _quantized_convolution_jvp(groups, primals, tangents):
    (x, kernel), (along_x, along_kernel) = primals, tangents
    held = (int8_round_trip(x, 0), int8_round_trip(kernel, 3))
    return (convolved(*held, groups),
            convolved(along_x, held[1], groups) + convolved(held[0], along_kernel, groups))


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


def test_a_quantized_convolutions_second_derivative_reads_its_scales():
    """The quantized convolution's gradient in its kernel, differentiated in its input (a mixed second
    derivative): the gradient reads the input quantized, whose scale (each example's absolute maximum)
    moves with the input, so the derivative goes through it, as the independent oracle's does
    (`quantized_convolution`): on par with the oracle against its float64 truth (check 1)."""
    from dew.nn.conv import Conv
    from dew.quantization import Quantization

    quantized = Quantization().apply(Conv(features=8, kernel_size=(3, 3), padding="SAME", use_bias=False))

    def mixed(convolution):
        def run(x, kernel, along_x, cotangent):
            def kernel_gradient(x):
                return jax.grad(lambda k: jnp.sum(convolution(x, k) * cotangent))(kernel)
            return jax.jvp(kernel_gradient, (x,), (along_x,))[1]
        return run

    def dew_convolution(x, kernel):
        return quantized.apply({"params": {"kernel": kernel}}, x)

    def oracle(x, kernel):
        return quantized_convolution(x, kernel, 1)

    def draw(seed):
        return tuple(normal_draw(2000 + seed, (1, 6, 6, 4), (3, 3, 4, 8), (1, 6, 6, 4), (1, 6, 6, 8)))

    assert_as_exact(jax.jit(mixed(dew_convolution)), jax.jit(mixed(oracle)), draw,
                    "a quantized convolution's mixed second derivative", seeds=8)


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
