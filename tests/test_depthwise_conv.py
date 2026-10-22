"""Depthwise 3x3 forward and VJP obey the fp32 reduction-order bound."""

import itertools
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from reference_error import assert_fp32_reduction_bound

from dew.nn.conv import (
    Conv,
    _conv_general_dilated,
    _materialized_depthwise_3x3,
    _polyphase_depthwise_3x3,
    _shifted_depthwise_3x3,
)
from dew.nn.ssm import SpatialFusionConv


def convolve(x, kernel, dilation):
    return jax.lax.conv_general_dilated(
        x, kernel, (1, 1), 'SAME', rhs_dilation=(dilation, dilation),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'), feature_group_count=x.shape[-1],
        precision=jax.lax.Precision.HIGHEST)


@pytest.mark.parametrize('operation', [_polyphase_depthwise_3x3, _shifted_depthwise_3x3,
                                       _materialized_depthwise_3x3])
@pytest.mark.parametrize('dilation', [1, 2, 3])
def test_depthwise_forward_and_gradients_keep_every_term_with_fp32_rounding(dilation, operation):
    """Two reductions differ by at most twice gamma_n times sum(abs(products)).

    Nine products contribute to each output and input gradient; B*H*W
    products contribute to each weight gradient. A missing corner moves the
    output by order one, far outside the bound even under cancellation; so
    does a grid interleaved back to the wrong pixels. The 7x8 image is not a
    multiple of either dilation, so the zero padding to a whole grid is
    checked too. At B16 16x16x768 on the RTX 4080 (JAX 0.11.2.post3) the
    fp32 output and dx were exact against the dilated convolution at
    highest precision, and dw was 0.0 of this bound; bf16 dw used 0.019 of
    it.
    """
    rng = np.random.default_rng(17)
    x = jnp.asarray(rng.normal(size=(2, 7, 8, 5)).astype(np.float32))
    kernel = jnp.asarray(rng.normal(size=(3, 3, 1, 5)).astype(np.float32))
    cotangent = jnp.asarray(rng.normal(size=x.shape).astype(np.float32))

    def forward_and_vjp(operation, x, kernel, cotangent):
        output, pullback = jax.vjp(operation, x, kernel)
        return output, *pullback(cotangent)

    reference = partial(convolve, dilation=dilation)
    candidate = partial(operation, dilation=dilation)
    expected = jax.jit(partial(forward_and_vjp, reference))(x, kernel, cotangent)
    actual = jax.jit(partial(forward_and_vjp, candidate))(x, kernel, cotangent)
    magnitudes = jax.jit(partial(forward_and_vjp, reference))(
        jnp.abs(x), jnp.abs(kernel), jnp.abs(cotangent))
    for got, want, magnitude, terms in zip(
            actual, expected, magnitudes, (9, 9, np.prod(x.shape[:-1])), strict=True):
        assert_fp32_reduction_bound(got, want, magnitude, terms)
    expected_loss = jnp.sum(expected[0] * cotangent)
    actual_loss = jnp.sum(actual[0] * cotangent)
    terms = x.size + 9
    loss_magnitude = np.sum(np.asarray(magnitudes[0], np.float64)
                            * np.abs(np.asarray(cotangent, np.float64)))
    assert_fp32_reduction_bound(actual_loss, expected_loss, loss_magnitude, terms)


@pytest.mark.parametrize('operation', [_polyphase_depthwise_3x3, _shifted_depthwise_3x3,
                                       _materialized_depthwise_3x3])
@pytest.mark.parametrize('dilation', [1, 2, 3])
def test_bf16_depthwise_accumulates_before_rounding(dilation, operation):
    """Rounding each add to bf16 loses eight half-ulp terms at an interior pixel."""
    side = 2 * dilation + 1
    x = jnp.ones((1, side, side, 2), jnp.bfloat16)
    kernel = jnp.full((3, 3, 1, 2), 1 / 256, jnp.bfloat16).at[1, 1].set(1)

    def forward_and_vjp(operation, x, kernel):
        output, pullback = jax.vjp(operation, x, kernel)
        return output, *pullback(jnp.ones_like(output))

    actual = jax.jit(partial(forward_and_vjp, partial(operation, dilation=dilation)))(x, kernel)
    expected = jax.jit(partial(forward_and_vjp, partial(convolve, dilation=dilation)))(x, kernel)
    for lhs, rhs in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(lhs, rhs)
    assert float(actual[0][0, dilation, dilation, 0]) == 1.03125
    assert float(actual[1][0, dilation, dilation, 0]) == 1.03125


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="the shifted sum is the CPU's path")
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize('dilation', [1, 2, 3])
@pytest.mark.parametrize('shape', [(2, 16, 16, 768), (1, 7, 8, 17), (1, 7, 8, 16)])
def test_cpu_depthwise_is_the_library_convolution_bit_for_bit(shape, dilation, dtype):
    """On the CPU a depthwise convolution of more than 16 features runs as
    nine shifted products summed in fp32, with no convolution op left, and
    its bits are the library convolution's (YNNPACK under jax 0.11.2.post3),
    so a sampled image does not change; 16 features or fewer, which the
    library sums otherwise, keep the convolution. The hybrid DiT's 2 x 16 x
    16 x 768 maps took 6.4 ms a call as a convolution and 1.4 ms shifted, on
    one core of an i9-12900K."""
    rng = np.random.default_rng(dilation)
    # A band of zero rows; every output keeps a nonzero tap. An output whose nine taps are all negative
    # zero is negative zero here, the sum's IEEE sign, where the library's sign follows its kernel.
    x = jnp.asarray(rng.normal(size=shape) * (np.arange(shape[1]) % 4 != 1)[None, :, None, None], dtype)
    kernel = jnp.asarray(rng.normal(size=(3, 3, 1, shape[-1])) * 0.2, dtype)
    padding = 'SAME' if dilation != 2 else ((2, 2), (2, 2))
    shared = jax.jit(lambda x, kernel: _conv_general_dilated(
        x, kernel, (1, 1), padding, rhs_dilation=(dilation, dilation),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'), feature_group_count=shape[-1]))
    bits = np.uint32 if dtype == jnp.float32 else np.uint16
    np.testing.assert_array_equal(np.asarray(shared(x, kernel)).view(bits), np.asarray(jax.jit(partial(
        convolve, dilation=dilation))(x, kernel)).view(bits))


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="the shifted sum is the CPU's path")
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize('dilation', [1, 2, 3])
def test_an_all_negative_zero_output_is_the_sums_negative_zero_where_the_librarys_sign_varies(
        dilation, dtype):
    """Channel 0 of a 1 x 7 x 8 x 17 map is zero over its first 2d + 1 rows and columns under negative
    taps, so the outputs whose nine taps all read it sum nine negative zeros. Dew's shifted sum gives
    the IEEE sum's negative zero there; XLA:CPU's library convolution (jax 0.11.2.post3) gives a zero
    of either sign, by its kernel and the host it runs on: positive at every dilation on one armada
    host, negative at dilations 1 and 2 on another (ledger 1361). Every other output is the library's
    bit for bit."""
    rng = np.random.default_rng(dilation)
    x, kernel = rng.normal(size=(1, 7, 8, 17)), rng.normal(size=(3, 3, 1, 17)) * 0.2
    x[:, :2 * dilation + 1, :2 * dilation + 1, 0] = 0
    kernel[..., 0] = -np.abs(kernel[..., 0])
    zero = np.zeros((1, 7, 8, 17), bool)
    for row, column in itertools.product(range(7), range(8)):
        taps = [(row + dilation * i, column + dilation * j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
        # A tap past the border reads the padding's zero; one inside reads the zeroed corner.
        zero[0, row, column, 0] = all(not (0 <= r < 7 and 0 <= c < 8) or max(r, c) <= 2 * dilation
                                      for r, c in taps)
    x, kernel = jnp.asarray(x, dtype), jnp.asarray(kernel, dtype)
    shared = np.asarray(jax.jit(lambda x, kernel: _conv_general_dilated(
        x, kernel, (1, 1), 'SAME', rhs_dilation=(dilation, dilation),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'), feature_group_count=17))(x, kernel), np.float32)
    library = np.asarray(jax.jit(partial(convolve, dilation=dilation))(x, kernel), np.float32)
    assert zero.any() and not shared[zero].any() and not library[zero].any()
    assert np.signbit(shared[zero]).all()
    np.testing.assert_array_equal(shared[~zero].view(np.uint32), library[~zero].view(np.uint32))



@pytest.mark.parametrize('dilation', [1, 2, 3])
@pytest.mark.parametrize('shape', [(7, 8, 4), (2, 7, 8, 4), (2, 1, 7, 8, 4)])
def test_shared_conv_preserves_flax_parameters_and_batch_dimensions(dilation, shape):
    rng = np.random.default_rng(4)
    x = jnp.asarray(rng.normal(size=shape).astype(np.float32))
    fields = {'features': 4, 'kernel_size': (3, 3), 'kernel_dilation': (dilation, dilation),
              'feature_group_count': 4, 'precision': jax.lax.Precision.HIGHEST}
    conv, reference = Conv(**fields), nn.Conv(**fields)
    variables = reference.init(jax.random.key(1), x)
    actual_variables = conv.init(jax.random.key(1), x)
    for actual, expected in zip(jax.tree.leaves(actual_variables), jax.tree.leaves(variables), strict=True):
        np.testing.assert_array_equal(actual, expected)
    variables['params']['bias'] = jnp.asarray(rng.normal(size=(4,)).astype(np.float32))
    magnitude = convolve(jnp.abs(x).reshape(-1, *shape[-3:]),
                         jnp.abs(variables['params']['kernel']), dilation).reshape(shape)
    magnitude = magnitude + jnp.abs(variables['params']['bias'])
    assert_fp32_reduction_bound(jax.jit(conv.apply)(variables, x),
                                jax.jit(reference.apply)(variables, x), magnitude, 10)


@pytest.mark.parametrize('changes', [
    {'feature_group_count': 2}, {'features': 8}, {'strides': 2},
    {'kernel_size': (5, 5)}, {'padding': 'VALID'}, {'kernel_dilation': (2, 3)},
])
def test_other_grouped_convolutions_keep_flax_numerics(changes):
    fields = {'features': 4, 'kernel_size': (3, 3), 'feature_group_count': 4,
              'precision': jax.lax.Precision.HIGHEST}
    fields.update(changes)
    x = jax.random.normal(jax.random.key(2), (2, 9, 8, 4))
    original, optimized = nn.Conv(**fields), Conv(**fields)
    variables = original.init(jax.random.key(1), x)
    np.testing.assert_array_equal(jax.jit(original.apply)(variables, x),
                                  jax.jit(optimized.apply)(variables, x))


def test_spatial_fusion_keeps_its_checkpoint_and_residual_add_order():
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(size=(2, 8, 7, 4)).astype(np.float32))
    kernels = {f'dwconv_dil{dilation}': {'kernel': jnp.asarray(
        rng.normal(size=(3, 3, 1, 4)).astype(np.float32))} for dilation in (1, 2, 3)}
    expected = x
    magnitude = jnp.abs(x)
    for dilation in (1, 2, 3):
        expected = expected + convolve(x, kernels[f'dwconv_dil{dilation}']['kernel'], dilation)
        magnitude = magnitude + convolve(
            jnp.abs(x), jnp.abs(kernels[f"dwconv_dil{dilation}"]["kernel"]), dilation
        )
    actual = jax.jit(SpatialFusionConv(4).apply)({'params': kernels}, x)
    assert_fp32_reduction_bound(actual, expected, magnitude, 28)


@pytest.mark.parametrize('operation', [_polyphase_depthwise_3x3, _materialized_depthwise_3x3])
def test_depthwise_boundaries_preserve_forward_and_higher_order_derivatives(operation):
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.normal(size=(1, 5, 6, 3)).astype(np.float32))
    kernel = jnp.asarray(rng.normal(size=(3, 3, 1, 3)).astype(np.float32))
    dx, dw = jnp.cos(x), jnp.sin(kernel)

    def derivatives(operation, x, kernel, dx, dw):
        def loss(x, kernel):
            return jnp.sum(operation(x, kernel) ** 2)

        return (jax.jvp(operation, (x, kernel), (dx, dw)),
                jax.jvp(jax.grad(loss, argnums=(0, 1)), (x, kernel), (dx, dw)))

    expected = jax.jit(partial(derivatives, partial(convolve, dilation=2)))(x, kernel, dx, dw)
    actual = jax.jit(partial(derivatives, partial(operation, dilation=2)))(x, kernel, dx, dw)
    for lhs, rhs in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(lhs, rhs, rtol=4e-6, atol=2e-5)


@pytest.mark.skipif(
    jax.default_backend() == "gpu",
    reason="Grouped int8/fp8 convolution is refused on GPU; test_quantization covers the refusal",
)
@pytest.mark.parametrize('training', [True, False])
def test_depthwise_quantization_keeps_the_original_provider_output(training):
    """QT and PTQ still reach the provider, with identical scales and output."""
    pytest.importorskip('qwix')
    from dew.quantization import Quantization, quantize_for_serving

    fields = {'features': 8, 'kernel_size': (3, 3), 'feature_group_count': 8,
              'use_bias': False, 'dtype': jnp.bfloat16}
    original = Conv(**fields, conv_general_dilated=jax.lax.conv_general_dilated)
    optimized = Conv(**fields)
    x = jax.random.normal(jax.random.key(2), (2, 7, 8, 8), jnp.bfloat16)
    variables = original.init(jax.random.key(1), x)
    if training:
        before = Quantization().apply(original).apply(variables, x)
        after = Quantization().apply(optimized).apply(variables, x)
    else:
        old, old_variables = quantize_for_serving(original, variables, Quantization(), x)
        new, new_variables = quantize_for_serving(optimized, variables, Quantization(), x)
        for lhs, rhs in zip(jax.tree.leaves(old_variables), jax.tree.leaves(new_variables), strict=True):
            np.testing.assert_array_equal(lhs, rhs)
        before, after = old.apply(old_variables, x), new.apply(new_variables, x)
    np.testing.assert_array_equal(before, after)


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="XLA:CUDA drops the zero (ledger 1361)")
def test_the_materialized_depthwise_sum_starts_from_a_positive_zero():
    """On the CPU the materialized form sums its nine taps onto a positive zero, so taps that are all
    negative zero (a zero input under a negative kernel) give positive zero, where the barriered form
    keeps the first product's, the sum's IEEE sign. XLA:CUDA drops the zero, and there both give
    negative zero (the H100 old-against-new run)."""
    x = jnp.zeros((1, 6, 6, 4), jnp.float32)
    w = -jnp.ones((3, 3, 1, 4), jnp.float32)
    for dilation in (1, 2):
        assert not np.signbit(np.asarray(_materialized_depthwise_3x3(x, w, dilation))).any()
        assert np.signbit(np.asarray(_shifted_depthwise_3x3(x, w, dilation))).all()

