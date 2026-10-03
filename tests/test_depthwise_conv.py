"""Depthwise 3x3 forward and VJP obey the fp32 reduction-order bound."""

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


@pytest.mark.parametrize('dtype, dilation, form', [
    (jnp.bfloat16, 2, 'polyphase'), (jnp.bfloat16, 3, 'polyphase'),
    (jnp.float32, 2, 'shifted'), (jnp.float32, 3, 'shifted'),
    (jnp.float32, 1, 'convolution'), (jnp.bfloat16, 1, 'convolution')])
def test_cuda_takes_the_dilated_depthwise_form_measured_faster_for_its_dtype(dtype, dilation, form):
    """On CUDA a dilated depthwise 3x3 convolution runs as the polyphase
    convolution in bf16 and as nine shifted products in fp32, the forms the
    hybrid DiT's step ran fastest in for each dtype on an RTX 4080; an
    undilated one is the convolution. Read off the CUDA lowering: the
    polyphase form convolves the dilation^2 grids along the batch, undilated,
    and the shifted one convolves nothing."""
    batch = 2
    x = jnp.zeros((batch, 12, 12, 32), dtype)
    kernel = jnp.zeros((3, 3, 1, 32), dtype)
    shared = jax.jit(lambda x, kernel: _conv_general_dilated(
        x, kernel, (1, 1), 'SAME', rhs_dilation=(dilation, dilation),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'), feature_group_count=32))
    text = shared.trace(x, kernel).lower(lowering_platforms=('cuda',)).as_text()
    convolutions = [line for line in text.splitlines() if 'stablehlo.convolution' in line]
    if form == 'shifted':
        assert not convolutions
    else:
        assert len(convolutions) == 1
        grids = batch * dilation ** 2 if form == 'polyphase' else batch
        assert f'tensor<{grids}x' in convolutions[0], convolutions[0]


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
    x = jnp.asarray(rng.normal(size=shape), dtype)
    kernel = jnp.asarray(rng.normal(size=(3, 3, 1, shape[-1])) * 0.2, dtype)
    padding = 'SAME' if dilation != 2 else ((2, 2), (2, 2))
    shared = jax.jit(lambda x, kernel: _conv_general_dilated(
        x, kernel, (1, 1), padding, rhs_dilation=(dilation, dilation),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'), feature_group_count=shape[-1]))
    np.testing.assert_array_equal(np.asarray(shared(x, kernel)), np.asarray(jax.jit(partial(
        convolve, dilation=dilation))(x, kernel)))
    assert (' convolution(' in shared.lower(x, kernel).compile().as_text()) == (shape[-1] <= 16)


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
    from dew.training.quantization import Quantization, quantize_for_serving

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


@pytest.mark.network
def test_published_hybrid_dit_samples_within_fp32_rounding(tmp_path):
    """Same 176M weights, prompt, seed 17 and 20 steps at highest precision.

    RTX 4080 / JAX 0.11.2.post3: max latent error 1.88053e-5 (RMS
    2.19756e-6), max fp32-decoded pixel error 3.99053e-5 (RMS 1.81749e-6).
    The 3e-5/5e-5 absolute bounds cover fp32 reduction-order changes in the
    20-step trajectory. Both paths disable TF32. The published bf16 decoder
    is recorded separately, with no image-identity assertion: rounding its
    activations maps independent max 1.88351e-5 / RMS 2.21749e-6 latent noise
    to max 0.05078125 / RMS 0.002173656 pixel differences, the same as the
    convolution change. The bf16 decoder's image sensitivity is recorded
    alongside the fp32 parity; it is outside an fp32 rounding bound.
    """
    from argparse import Namespace

    from tools.benchmark_depthwise import checkpoint

    checkpoint(Namespace(output=tmp_path / 'checkpoint.json'))
