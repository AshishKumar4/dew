"""Float-range and nested-derivative contracts of the products' seams: embedding lookups, bf16 and wide CPU
products, the grouped GEMM and its portable projection, the GEMM prologue and transposed hooks, a stored
expert primal, the frozen FP8 product, the quantized convolution, a broadcast summing its gradient wide
(ledger 2641; the harness is tests/contract_support.py)."""

import functools
import itertools

import jax
import jax.numpy as jnp
import pytest
from contract_support import EDGES, FORWARD_ORDERS, OWN_FORWARD_ORDERS, assert_contract, normal_draw

from dew.interop import torchax_fallback
from dew.nn import precision
from dew.nn.blocks import table_rows
from dew.nn.kernels.templates.gemm import template


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
    """The GEMM template's grouped product (`template.gemm`, its custom derivative through
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
        return template.gemm(rows, matrices, sizes, out_dtype=rows.dtype)

    def plain(rows, matrices):
        return jnp.einsum("rg,rk,gkn->rn", jax.nn.one_hot(groups, 3, dtype=rows.dtype), rows, matrices)

    assert_contract(grouped, primals, direction, cotangent, f"a grouped GEMM {edge}", truth_fn=plain,
                    plain_fn=plain)


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


def rms_producer(lhs, scale):
    """An RMS norm as a GEMM's producer writes it: the operand over its root mean square, times its scale."""
    return lhs * jax.lax.rsqrt(jnp.mean(jnp.square(lhs), axis=-1, keepdims=True) + 1e-6) * scale


@pytest.mark.parametrize("edge", EDGES)
def test_a_gemm_prologue_holds_every_order(edge):
    """The GEMM template's prologue (`template._prologued`, a custom JVP whose tangent is the producer's own,
    as the fused GEMM hands it, `fused._whole_jvp`): every order finite wherever float64 is a normal float32
    and the producer as written is."""
    from dew.nn.kernels.templates.gemm.template import _prologued

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
def test_a_transposed_hooks_custom_derivative_holds_every_order_it_takes(edge):
    """A custom derivative within a hook a mainloop runs on transposed tiles (`layout.transposed`, its rule
    held entry by entry, `_custom.held`): every order it takes finite wherever float64 is a normal float32
    and the hook on the tile as its caller holds it is; it takes no forward-derivative order, under a nested
    jit and a scan too (ledger 2658)."""
    from dew.nn.kernels.templates import layout

    (tile,) = normal_draw(105, (6, 4), scale=EDGES[edge])
    (direction,) = normal_draw(106, (6, 4), scale=EDGES[edge])
    (cotangent,) = normal_draw(107, (6, 4))

    def plain(tile):
        return softly_rectified(tile.T).T

    assert_contract(layout.transposed(softly_rectified), (tile,), (direction,), cotangent,
                    f"a transposed hook's custom derivative {edge}", truth_fn=plain, plain_fn=plain,
                    without=FORWARD_ORDERS)


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


@pytest.mark.parametrize("edge", [edge for edge in LINEAR_EDGES if edge != "near the largest"])
def test_a_frozen_fp8_product_holds_every_order(edge):
    """A frozen FP8 kernel's product (`quantization.frozen_fp8_product`, its tangent the decoded kernel's):
    every order in the input finite wherever float64 is a normal float32 and the product with the kernel its
    codes decode to (each row's codes times its scale) is, the kernel a unit draw and the input at
    `LINEAR_EDGES`, so the product itself nears
    float32's ends. Near the largest its input's scale overflows before the kernel's applies (ledger 2715),
    that row's head adds the edge."""
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
    """A quantized convolution under Qwix's quantized training, one group and two (`quantization`'s grouped
    convolution, a custom VJP whose backward is the float convolution's transpose at the dequantized
    operands): every order it takes finite wherever float64 is a normal float32 and the convolution of the
    operands quantized to int8 and back (each example's scale, each output feature's; a grouped input first
    over each example's peak in each group, restored on that group's outputs) is, its reverse mode the float
    convolution's transpose at the dequantized operands, as the seam declares, and a derivative of that
    through the quantization, the scales' included (`quantized_convolution`); it takes no
    forward derivative of itself, nor a second (ledger 2658), every forward derivative of its gradient
    running."""
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
                    truth_fn=plain, plain_fn=plain, without=OWN_FORWARD_ORDERS)


@pytest.mark.parametrize("edge", EDGES)
def test_a_broadcast_summing_its_gradient_wide_holds_every_order(edge):
    """`torchax_fallback._accumulated` (a broadcast whose gradient sums in float32 and rounds once to its
    value's dtype, as torch's autograd reduces a bf16 gradient): every order runs, finite wherever float64
    is a normal float32, at values near float32's ends, beside the plain broadcast; but forward mode,
    which a custom_vjp takes none of, the orders ledger 2653 adds with its fix."""
    value, direction = normal_draw(959, (3, 1, 4), (3, 1, 4), scale=EDGES[edge])
    (cotangent,) = normal_draw(961, (3, 5, 4))

    def broadcast(value):
        return torchax_fallback._accumulated(value, (3, 5, 4), (0, 1, 2))

    def plain(value):
        return jnp.broadcast_to(value, (3, 5, 4))

    # Forward mode's orders (`orders`), which ledger 2653 lists.
    forward = ("jvp", "jvp of grad", "second jvp", "jvp of grad, remat", "second jvp, remat",
               "jvp of grad, nested jit", "jvp of grad, scanned")
    assert_contract(broadcast, (value,), (direction,), cotangent, f"a wide-summed broadcast {edge}",
                    truth_fn=plain, plain_fn=plain, without=forward)
