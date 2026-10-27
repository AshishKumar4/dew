"""Quantized training through Qwix: the value, the wrapping, the training.

Qwix-dependent tests skip without the package; the value's refusals run
everywhere, since construction never imports it.
"""

import dataclasses
import functools
import itertools
import os
import subprocess
import sys
import types
from collections.abc import Callable
from importlib import import_module
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from flax.core import FrozenDict
from reference_error import (
    as_given,
    assert_as_exact,
    assert_as_exact_as_the_reference,
    assert_fp32_reduction_bound,
    with_gradients,
)
from token_support import token_rows

from dew.config import ObjectiveConfig
from dew.distributed.batches import shard_batch
from dew.distributed.placement import Layout, MeshSpec
from dew.lora import _updated as lora_update
from dew.nn.attention_residuals import depth_mixed
from dew.nn.blocks import torch_bicubic_resize
from dew.nn.kv_cache import hadamard, rotated
from dew.nn.precision import FP32_SUMS
from dew.nn.quantizers import nearest_codewords
from dew.nn.sharding import pipeline_microbatches
from dew.nn.ssm import diagonal_recurrence
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.optim import OptimConfig
from dew.quantization import UNQUANTIZED, Quantization, quantize_for_serving, quantize_with
from dew.registry import models

import_module("dew.nn.backbones")  # registers the fixture kind


VOCAB = 64
SEQ_LEN = 8
BATCH = 4
TINY = {"vocab_size": VOCAB, "emb_features": 32, "num_layers": 2, "num_heads": 4,
            "num_kv_heads": 2, "mlp_features": 64, "max_seq_len": 16}


def tiny(**overrides):
    return models.build("causal_transformer", **{**TINY, **overrides})


def token_batch():
    return token_rows(VOCAB, BATCH, SEQ_LEN + 1)


def quantized_forward(spec, **overrides):
    model = tiny(**overrides)
    variables = model.init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    qmodel = spec.apply(model)
    ids = jnp.asarray(token_batch()["text"][:, :SEQ_LEN])
    return model, qmodel, variables, ids


@pytest.mark.parametrize("scan_layers", [False, True])
def test_the_wrapped_forward_matches_qwixs_own_call(scan_layers):
    """Dew's wrapping against the provider built by hand, over the layer
    loop and over the scanned stack, the hand-built one traced under its own key (`quantize_with`),
    as Dew traces every Qwix-quantized model so Qwix owns its products: identical compiled logits, and both
    away from the fp32 forward. The gap is the assertion: rules that never
    reached a matmul would leave the fp32 numerics bitwise, so any bit moved
    says they did. Observed with both layouts: wrapped and hand-built bitwise
    equal on CPU and on the RTX 4080, 1.2e-01 (CPU) and 1.3e-01 (RTX 4080)
    from fp32 on logits up to 3.9."""
    qwix = pytest.importorskip("qwix")
    model, qmodel, variables, ids = quantized_forward(Quantization(), scan_layers=scan_layers)
    rules = [qwix.QtRule(module_path=".*", weight_qtype=jnp.int8,
                         act_qtype=jnp.int8)]
    reference = quantize_with(model, qwix.QtProvider(rules))
    plain = jax.jit(model.apply)(variables, ids)
    wrapped = jax.jit(qmodel.apply)(variables, ids)
    manual = jax.jit(reference.apply)(variables, ids)
    np.testing.assert_array_equal(wrapped, manual)
    assert not np.array_equal(wrapped, plain)


def test_a_dense_qwix_quantizes_directly_takes_its_gradient_on_a_cpu():
    """A Dew Dense Qwix quantizes directly (`qwix.quantize_model`, outside `quantize_with`'s trace)
    differentiates on a CPU: its product calls a backward of its own (a custom_vjp), so the float64
    sums (`precision.accumulated`) leave it to Qwix, forward and backward, as on any platform."""
    qwix = pytest.importorskip("qwix")
    from dew.nn.dense import Dense

    layer = Dense(8)
    with jax.default_device(jax.devices("cpu")[0]):
        x = jnp.asarray(np.random.default_rng(0).standard_normal((4, 16)), jnp.float32)
        variables = layer.init(jax.random.key(0), x)
        quantized = qwix.quantize_model(layer, qwix.QtProvider(
            [qwix.QtRule(module_path=".*", weight_qtype=jnp.int8, act_qtype=jnp.int8)]))
        loss = jax.grad(lambda variables: jnp.sum(quantized.apply(variables, x) ** 2))
        gradients = jax.jit(loss)(variables)
    kernel = np.asarray(gradients["params"]["kernel"])
    assert np.all(np.isfinite(kernel)) and np.any(kernel)


def test_quantized_and_plain_calls_of_one_shape_keep_their_own_products_in_one_process():
    """`x @ w` is `jnp.matmul`, a jitted function whose trace a quantized call takes under Qwix's
    interception. Beside the plain model at the same shapes, in either order and each shape's first
    call cold, each computes its own product: a plain call after a quantized one is the plain
    product (`lax.dot_general`'s, which no jitted contraction serves), and a quantized call after a
    plain one quantizes."""
    pytest.importorskip("qwix")

    class Product(nn.Module):
        @nn.compact
        def __call__(self, x):
            return x @ self.param("kernel", nn.initializers.normal(1.0), (x.shape[-1], 8))

    model = Product()
    quantized = Quantization().apply(model)
    variables = model.init(jax.random.key(0), jnp.ones((1, 16)))
    kernel = variables["params"]["kernel"]

    def plain(x):
        return np.asarray(jax.lax.dot_general(x, kernel, (((1,), (0,)), ((), ()))))

    first, then = (jax.random.normal(jax.random.key(seed), (rows, 16)) for seed, rows in ((1, 4), (2, 5)))
    assert not np.array_equal(np.asarray(quantized.apply(variables, first)), plain(first))
    np.testing.assert_array_equal(np.asarray(model.apply(variables, first)), plain(first))
    np.testing.assert_array_equal(np.asarray(model.apply(variables, then)), plain(then))
    assert not np.array_equal(np.asarray(quantized.apply(variables, then)), plain(then))


def test_qwix_refuses_an_unknown_calibration_where_it_first_quantizes_a_weight():
    """Dew keeps no copy of Qwix's calibration grammar: a spec with a method Qwix does not know builds,
    and the first forward that quantizes a matched weight raises, Qwix's own refusal."""
    pytest.importorskip("qwix")
    _, qmodel, variables, ids = quantized_forward(Quantization(calibration="not_a_method"))
    with pytest.raises(ValueError):
        qmodel.apply(variables, ids)


def test_a_pattern_that_matches_nothing_quantizes_nothing():
    """A narrowed pattern is selective: a regex no module path matches runs
    the fp32 numerics bitwise, as the unquantized model sums them where Qwix
    owns its products (`FP32_SUMS`). A pattern that matched everything would
    move the logits here, so the equality fails it."""
    pytest.importorskip("qwix")
    model, qmodel, variables, ids = quantized_forward(
        Quantization(patterns=(".*does_not_exist.*",)))
    with jax.default_matmul_precision(FP32_SUMS):
        plain = model.apply(variables, ids)
    assert float(jnp.max(jnp.abs(qmodel.apply(variables, ids) - plain))) == 0.0


def test_an_int8_trunk_trains_down():
    """Five adamw steps through the int8 trunk: each loss below the one
    before, which quantized gradients pointing the wrong way or drowned in
    rounding noise would not manage. Observed on CPU: 5.17 down to 4.07 in
    steps of about 0.25, against 1e-7 of int8 rounding."""
    pytest.importorskip("qwix")
    _, qmodel, variables, _ = quantized_forward(Quantization())
    objective = LMObjective(qmodel, SEQ_LEN)
    batch = token_batch()
    solver = OptimConfig(learning_rate=1e-3).build(5)
    params, opt_state = variables["params"], solver.init(variables["params"])

    @jax.jit
    def train(params, opt_state, key):
        (loss, _), grads = jax.value_and_grad(
            lambda p: objective.scalar_loss({**variables, "params": p}, batch,
            Step(step=jnp.zeros((), jnp.int32), key=key, ema=None)),
            has_aux=True)(params)
        updates, opt_state = solver.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    losses = []
    key = jax.random.key(0)
    for _ in range(5):
        key, subkey = jax.random.split(key)
        params, opt_state, loss = train(params, opt_state, subkey)
        losses.append(float(loss))
    assert all(later < earlier for earlier, later in itertools.pairwise(losses)), losses


CONVOLUTION_SEEDS = 64
"""The quantized convolutions' declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for`
on armada."""


def ungrouped_reference(qwix, settings: dict, group_width: int):
    """Qwix's own quantized training of a grouped convolution written ungrouped, as a function of the
    grouped kernel, the input and its group peaks (`peaks_of`, returned beside it): the kernel
    block-diagonal, zero across groups, the input divided by the peaks as Dew divides it and the output
    scaled back. The zeros quantize to zero and every group then peaks at 1, so the two quantize to the
    same integers. The peaks are an argument, so a derivative holds them fixed, as Dew's does."""
    from dew.nn.conv import Conv

    features = settings["features"]
    ungrouped = quantize_with(Conv(**settings), qwix.QtProvider(
        [qwix.QtRule(module_path=".*", weight_qtype=jnp.int8, act_qtype=jnp.int8)]))
    # Output channel o reads input channel i as its group's j-th: block[j, o, i].
    block = (jnp.arange(features)[None, None, :]
             == (jnp.arange(features)[None, :, None] // group_width) * group_width
             + jnp.arange(group_width)[:, None, None])

    def peaks_of(x):
        rows = x.reshape(*x.shape[:-1], features // group_width, group_width)
        peaks = jnp.max(jnp.abs(rows), axis=(1, 2, 4))
        return jnp.repeat(peaks, group_width, axis=1)[:, None, None, :]

    def reference(kernel, x, peaks):
        block_diagonal = jnp.einsum("hwjo,joi->hwio", kernel, block.astype(kernel.dtype))
        return ungrouped.apply({"params": {"kernel": block_diagonal}}, x / peaks) * peaks
    return reference, peaks_of


@pytest.mark.skipif(jax.default_backend() == "gpu",
                    reason="Dew refuses a grouped quantized convolution on a GPU; "
                           "test_a_gpu_refuses_grouped_quantized_convolutions covers it")
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("group_width", [1, 4])
def test_a_grouped_convolution_quantizes_each_group_on_its_own_range(group_width, dtype):
    """A grouped convolution never adds one group's inputs into another's
    outputs, so each group's activations take their own int8 scale. Over 64
    channels whose ranges span 1e-3 to 1, depthwise and 4 channels per
    group, in float32 and in bf16 compute, the quantized convolution is on
    par (check 1) with Qwix's own quantization of it written ungrouped on
    each group's range (`ungrouped_reference`), from the float64
    convolution of the same inputs and kernel, each output channel its own
    leaf, so a quiet group is judged on its own and not pooled with the loud
    ones; and quantization moves every channel off the float convolution.
    One scale per example for all channels rounds the small groups to zero
    instead: observed on CPU before the fix, the worst channel 100%
    (depthwise) and 107% (4 per group) off, where the per-group quantization
    is 1.0% to 1.4% off."""
    qwix = pytest.importorskip("qwix")
    from dew.nn.conv import Conv

    features = 64
    settings = {"features": features, "kernel_size": (3, 3), "padding": "SAME", "use_bias": False}
    conv = Conv(**settings, feature_group_count=features // group_width, dtype=dtype)
    reference, peaks_of = ungrouped_reference(qwix, {**settings, "dtype": dtype}, group_width)
    ranges = jnp.logspace(-3, 0, features)
    init = jax.jit(conv.init)
    quantized = Quantization().apply(conv)

    def draw(seed):
        x = (jax.random.normal(jax.random.key(2 * seed), (2, 8, 8, features)) * ranges).astype(dtype)
        kernel = init(jax.random.key(2 * seed + 1), x)["params"]["kernel"].astype(dtype)
        return x, kernel

    def channels(output):
        """Each output channel its own leaf, `[B, H, W]`."""
        return {f"channel {index}": output[..., index] for index in range(features)}

    def dew(x, kernel):
        return channels(quantized.apply({"params": {"kernel": kernel}}, x))

    def plain(x, kernel):
        return channels(conv.clone(dtype=jnp.float64).apply({"params": {"kernel": kernel}}, x))

    x, kernel = draw(0)
    found = quantized.apply({"params": {"kernel": kernel}}, x)
    assert found.dtype == dtype
    unquantized = conv.apply({"params": {"kernel": kernel}}, x)
    for index in range(features):
        assert not np.array_equal(found[..., index], unquantized[..., index]), index
    qwix_own = jax.jit(lambda x, kernel: channels(reference(kernel, x, peaks_of(x))))
    label = f"grouped {group_width} {jnp.dtype(dtype).name} quantized convolution"
    assert_as_exact(jax.jit(dew), qwix_own, draw, label,
                    seeds=CONVOLUTION_SEEDS, truth_fn=plain)


@pytest.mark.skipif(jax.default_backend() == "gpu",
                    reason="Dew refuses a grouped quantized convolution on a GPU; "
                           "test_a_gpu_refuses_grouped_quantized_convolutions covers it")
@pytest.mark.parametrize("groups", [1, 16])
def test_a_quantized_convolution_keeps_its_calls_result_dtype_whatever_precision_it_pins(groups):
    """A quantized convolution keeps its call's contract, as a quantized matmul does: one that asks
    for a float32 result from bf16 operands returns float32, where before it returned the bf16
    input's dtype, on par (check 1) with Qwix's own quantization of the convolution on each group's
    range (`ungrouped_reference`) from the float64 convolution over drawn inputs and kernels; and one
    pinned at the highest precision, Dew's Conv and lax's alike, quantizes as the same call unpinned
    does, bit for bit, a quantized product having no precision to honour. Each is not the float
    convolution: quantization reached it."""
    from flax import linen as nn

    from dew.nn.conv import Conv

    class Native(nn.Module):
        """A convolution written as a lax call, asking for its own precision and result dtype."""
        precision: jax.lax.Precision | None = None
        preferred: jnp.dtype | None = None

        @nn.compact
        def __call__(self, x):
            kernel = self.param("kernel", nn.initializers.lecun_normal(), (3, 3, 16 // groups, 16))
            return jax.lax.conv_general_dilated(
                x, kernel.astype(x.dtype), (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"),
                feature_group_count=groups, precision=self.precision, preferred_element_type=self.preferred)

    x = jax.random.normal(jax.random.key(0), (2, 8, 8, 16)).astype(jnp.bfloat16)
    settings = {"features": 16, "kernel_size": (3, 3), "padding": "SAME", "feature_group_count": groups,
                "use_bias": False, "dtype": jnp.bfloat16}
    variables = Native().init(jax.random.key(1), x)

    def quantized(module):
        return Quantization().apply(module).apply(variables, x)

    for pinned, plain in ((Conv(**settings, precision=jax.lax.Precision.HIGHEST), Conv(**settings)),
                          (Native(precision=jax.lax.Precision.HIGHEST), Native())):
        found = quantized(pinned)
        np.testing.assert_array_equal(found, quantized(plain))
        assert not np.array_equal(found, plain.apply(variables, x))
    wide = Native(preferred=jnp.float32)
    found = quantized(wide)
    assert found.dtype == jnp.float32
    assert not np.array_equal(found, wide.apply(variables, x))
    qwix = pytest.importorskip("qwix")
    reference, peaks_of = ungrouped_reference(qwix, {key: value for key, value in settings.items()
                                                     if key != "feature_group_count"}, 16 // groups)
    widened = Quantization().apply(wide)

    def draw(seed):
        x = jax.random.normal(jax.random.key(2 * seed), (2, 8, 8, 16)).astype(jnp.bfloat16)
        return x, wide.init(jax.random.key(2 * seed + 1), x)["params"]["kernel"].astype(jnp.bfloat16)

    def plain(x, kernel):
        return jax.lax.conv_general_dilated(x, kernel, (1, 1), "SAME", feature_group_count=groups,
                                            dimension_numbers=("NHWC", "HWIO", "NHWC"))

    assert_as_exact(jax.jit(lambda x, kernel: widened.apply({"params": {"kernel": kernel}}, x)),
                    jax.jit(lambda x, kernel: reference(kernel, x, peaks_of(x))), draw,
                    f"float32 result of {groups} groups", seeds=CONVOLUTION_SEEDS, truth_fn=plain)


@pytest.mark.skipif(jax.default_backend() == "gpu",
                    reason="Dew refuses a grouped quantized convolution on a GPU; "
                           "test_a_gpu_refuses_grouped_quantized_convolutions covers it")
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("group_width", [1, 4])
def test_a_quantized_grouped_convolution_differentiates_as_qwix_does_ungrouped(group_width, dilation):
    """Quantized training differentiates a grouped convolution: its value
    and its gradients for the kernel and the input are those of Qwix's own
    quantized training of the same convolution written ungrouped, with a
    block-diagonal kernel, on the input divided by its group peaks as Dew
    divides it. A grouped convolution is an ungrouped one whose kernel is
    zero across groups, the zeros quantize to zero, and every group then
    peaks at 1, so the two quantize to the same integers and Qwix's
    ungrouped backward is the reference. The values are bitwise equal. The
    gradients sum the same nonzero products in other orders (the reference's
    zeros add exactly), so each entry differs by at most float32's bound for
    two reductions of those products (`assert_fp32_reduction_bound`): 9 per
    group channel and one division by the peak for the input's, one per
    input position for the kernel's. Observed on CPU at the highest
    precision: within 2.2e-07 relative, and 0.5% to 0.9% from the float
    convolution's gradients, inside int8's 2%. Before, Qwix's grouped
    backward raised (`128 // 128 != 128` for the hybrid DiT's depthwise
    convolutions)."""
    qwix = pytest.importorskip("qwix")
    from qwix._src.core import conv_general, qarray

    from dew.nn.conv import Conv

    features = 64
    groups = features // group_width
    settings = {"features": features, "kernel_size": (3, 3), "padding": "SAME",
                "kernel_dilation": (dilation, dilation), "use_bias": False}
    grouped = Conv(**settings, feature_group_count=groups)
    grouped_reference, peaks_of = ungrouped_reference(qwix, settings, group_width)
    x = jax.random.normal(jax.random.key(0), (2, 8, 8, features)) * jnp.logspace(-3, 0, features)
    kernel = grouped.init(jax.random.key(1), x)["params"]["kernel"]
    cotangent = jax.random.normal(jax.random.key(2), x.shape)
    # The peaks are the input's own, held fixed as the derivative is taken, as Dew holds them.
    peaks = peaks_of(x)

    def dew(kernel, x):
        return Quantization().apply(grouped).apply({"params": {"kernel": kernel}}, x)

    def reference(kernel, x):
        return grouped_reference(kernel, x, peaks)

    def plain(kernel, x):
        return grouped.apply({"params": {"kernel": kernel}}, x)

    def value_and_gradients(function):
        out, transpose = jax.vjp(jax.jit(function), kernel, x)
        return (out, *transpose(cotangent))

    def dequantized(array, for_lhs):
        how = conv_general.get_how_to_quantize(
            dimension_numbers=jax.lax.conv_dimension_numbers(x.shape, kernel.shape, ("NHWC", "HWIO", "NHWC")),
            for_lhs=for_lhs, qtype=jnp.int8, calibration_method="absmax")
        return qarray.dequantize(qarray.quantize(array, how))

    def convolve(x, kernel):
        return jax.lax.conv_general_dilated(
            x,
            kernel,
            (1, 1),
            "SAME",
            rhs_dilation=(dilation, dilation),
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=groups,
        )

    with jax.default_matmul_precision("highest"):
        got, want, float_ = (value_and_gradients(function) for function in (dew, reference, plain))
        # The absolute products each gradient sums: the dequantized operands
        # the backward reads, and the cotangent scaled by the peaks.
        _, transpose = jax.vjp(
            convolve,
            jnp.abs(dequantized(x / peaks, for_lhs=True)),
            jnp.abs(dequantized(kernel, for_lhs=False)),
        )
        x_magnitude, kernel_magnitude = transpose(jnp.abs(cotangent * peaks))
    np.testing.assert_array_equal(got[0], want[0])
    assert_fp32_reduction_bound(got[1], want[1], kernel_magnitude, int(np.prod(x.shape[:-1])))
    assert_fp32_reduction_bound(got[2], want[2], x_magnitude / peaks, 9 * group_width + 1)
    for got_part, float_part in zip(got[1:], float_[1:], strict=True):
        assert not np.array_equal(got_part, float_part)


QUANTIZED_CONVOLUTION = """
import sys

import jax
import jax.numpy as jnp
import numpy as np
import qwix

from dew.nn.conv import Conv
from dew.quantization import Quantization, quantize_with

dtype = sys.argv[1]
# Without the bias, which the quantization leaves alone and a bf16
# module adds, and differentiates, in bf16.
settings = {"features": 128, "kernel_size": (2, 2), "strides": (2, 2), "padding": "VALID", "use_bias": False}
conv = Conv(**settings, dtype=jnp.bfloat16)
x = jax.random.normal(jax.random.key(0), (8, 16, 16, 4), jnp.bfloat16)
variables = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), conv.init(jax.random.key(1), x))
cotangent = jax.random.normal(jax.random.key(2), (8, 8, 8, 128), jnp.bfloat16)
qtype = jnp.int8 if dtype == "int8" else jnp.float8_e4m3fn
reference = quantize_with(Conv(**settings), qwix.QtProvider(
    [qwix.QtRule(module_path=".*", weight_qtype=qtype, act_qtype=qtype)]))


def dew(variables, x):
    return Quantization(dtype=dtype).apply(conv).apply(variables, x)


def qwix_in_float32(variables, x):
    as_float32 = jax.tree.map(lambda leaf: leaf.astype(jnp.float32), (variables, x))
    return reference.apply(*as_float32).astype(jnp.bfloat16)


def value_and_gradients(function):
    out, transpose = jax.vjp(jax.jit(function), variables, x)
    return out, *transpose(cotangent)


got, want = value_and_gradients(dew), value_and_gradients(qwix_in_float32)
assert got[0].dtype == jnp.bfloat16
jax.tree.map(np.testing.assert_array_equal, got, want)
"""


@pytest.mark.parametrize("dtype", ["int8", "fp8"])
def test_a_bf16_quantized_convolution_trains_as_qwix_does_in_float32(dtype):
    """A bf16 convolution with quantized activations, a hybrid DiT's patch
    embedding, computes in quantized training what Qwix's own quantized
    training computes on float32 copies of its bf16 input and kernel,
    rounded to bf16: the same value and gradients, bitwise, compiled, on
    CPU and on the RTX 4080. Before, Qwix scaled the bf16 module's 8-bit
    product in bf16, and XLA:GPU failed to compile the int8 convolution
    (`UNIMPLEMENTED: Can't lower one or more integer convolutions`).

    It runs in a fresh process under the lane's flags without
    `--xla_gpu_autotune_level=0`, as a training run compiles: at level 0
    XLA leaves a cuDNN call the configuration it was emitted with, which
    cuDNN refuses for the fp8 convolution on an H100
    (CUDNN_STATUS_NOT_SUPPORTED), and XLA reads its flags once a process."""
    pytest.importorskip("qwix")
    flags = [flag for flag in os.environ.get("XLA_FLAGS", "").split()
             if not flag.startswith("--xla_gpu_autotune_level")]
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run([sys.executable, "-c", QUANTIZED_CONVOLUTION, dtype], capture_output=True, text=True,
                         env={**os.environ, "XLA_FLAGS": " ".join(flags), "PYTHONPATH": str(root / "src")},
                         timeout=600)
    assert run.returncode == 0, run.stderr[-3000:]


def test_a_bf16_convolution_under_weight_only_quantization_trains_as_qwix_does():
    """Weight-only quantized training keeps convolutions in float
    (`Quantization.weight_only`), so a bf16 convolution computes in bf16
    what Qwix's own quantized training with the same weight-only rule
    computes: value and gradients bitwise equal, compiled, on CPU and on the
    RTX 4080. Only a convolution with quantized activations quantizes from
    float32."""
    qwix = pytest.importorskip("qwix")
    from dew.nn.conv import Conv

    conv = Conv(features=128, kernel_size=(2, 2), strides=(2, 2), padding="VALID", use_bias=False,
                dtype=jnp.bfloat16)
    x = jax.random.normal(jax.random.key(0), (8, 16, 16, 4), jnp.bfloat16)
    variables = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), conv.init(jax.random.key(1), x))
    cotangent = jax.random.normal(jax.random.key(2), (8, 8, 8, 128), jnp.bfloat16)
    reference = quantize_with(conv, qwix.QtProvider([qwix.QtRule(
        module_path=".*", weight_qtype=jnp.int8, op_names=("dot_general", "einsum", "dot"))]))

    def value_and_gradients(module):
        out, transpose = jax.vjp(jax.jit(module.apply), variables, x)
        return out, *transpose(cotangent)

    got = value_and_gradients(Quantization(weight_only=True).apply(conv))
    assert got[0].dtype == jnp.bfloat16
    jax.tree.map(np.testing.assert_array_equal, got, value_and_gradients(reference))
    jax.tree.map(np.testing.assert_array_equal, got, value_and_gradients(conv))


@pytest.mark.parametrize("kernels", ["initialized", "trained"])
def test_a_convolution_no_rule_quantizes_computes_as_unwrapped_and_as_qwix_does(kernels):
    """The hybrid DiT's spatial fusion (depthwise, dilations 1 to 3), left out
    of the patterns, computes under Dew's quantized training as the
    unwrapped model does and as Qwix's own quantized training does, and
    serves as the unwrapped model does: value and gradients bitwise equal,
    compiled, at the zero-initialized kernels training starts from and at
    trained ones. Dew's Conv computes a dilated depthwise convolution as
    shifted products on CUDA, and before, Dew's provider sent an excluded one
    through lax, so the kernels' first gradients differed and quantized
    training of the hybrid DiT parted from Qwix's own after its first update
    (the RTX 4080; on CPU the input's gradient differed by 4.8e-07)."""
    qwix = pytest.importorskip("qwix")
    from flax import linen as nn

    from dew.nn.ssm import SpatialFusionConv

    class Block(nn.Module):
        @nn.compact
        def __call__(self, x):
            return SpatialFusionConv(64, name="spatial_fusion")(x)

    block = Block()
    x = jax.random.normal(jax.random.key(0), (2, 8, 8, 64))
    variables = block.init(jax.random.key(1), x)
    if kernels == "trained":
        variables = jax.tree.map(
            lambda leaf: 0.1 * jax.random.normal(jax.random.key(2), leaf.shape), variables
        )
    cotangent = jax.random.normal(jax.random.key(3), x.shape)
    pattern = "^(?!.*spatial_fusion).*"
    spec = Quantization(patterns=(pattern,))
    direct = quantize_with(block, qwix.QtProvider(
        [qwix.QtRule(module_path=pattern, weight_qtype=jnp.int8, act_qtype=jnp.int8)]))

    def value_and_gradients(module, variables):
        out, transpose = jax.vjp(jax.jit(module.apply), variables, x)
        return out, *transpose(cotangent)

    plain = value_and_gradients(block, variables)
    jax.tree.map(np.testing.assert_array_equal, value_and_gradients(spec.apply(block), variables), plain)
    jax.tree.map(np.testing.assert_array_equal, value_and_gradients(direct, variables), plain)
    served, served_variables = quantize_for_serving(block, variables, spec, x)
    np.testing.assert_array_equal(jax.jit(served.apply)(served_variables, x), plain[0])


def test_a_recorded_kernel_no_rule_matches_multiplies_by_its_own_recipe():
    """A served kernel stored quantized that reaches a matmul under rules naming none of it, as an unpacked
    task's member does under a rule naming its packed group, multiplies by the recipe it records: its
    weight-only logits bit for bit those of the rules that quantized it. Rules that quantize activations
    refuse it, whose recipe the kernel does not record."""
    pytest.importorskip("qwix")
    model, _, variables, ids = quantized_forward(Quantization(weight_only=True, patterns=(".*_proj",)))
    quantized = Quantization(weight_only=True, patterns=(".*_proj",))
    served, served_variables = quantize_for_serving(model, variables, quantized, ids)
    nothing = Quantization(weight_only=True, patterns=("none",))
    unmatched, _ = quantize_for_serving(model, variables, nothing, ids)
    np.testing.assert_array_equal(jax.jit(unmatched.apply)(served_variables, ids),
                                  jax.jit(served.apply)(served_variables, ids))
    activations, _ = quantize_for_serving(model, variables, Quantization(patterns=("none",)), ids)
    with pytest.raises(ValueError):
        activations.apply(served_variables, ids)


def test_serving_stores_int8_kernels_and_computes_what_training_quantized():
    """Served weights are int8 values with their scales, and the served
    projections reproduce the quantized-training forward they were trained
    under bit for bit, away from fp32 (unquantized products would leave it
    bitwise). Observed on CPU: 1.2e-01 from fp32 on logits of order 3.
    (Qwix's serving and training providers quantize attention's matmuls of
    two activations differently: with those quantized too, the two forwards
    differ by 3e-02.)"""
    pytest.importorskip("qwix")
    spec = Quantization(patterns=(".*_proj",))
    model, qmodel, variables, ids = quantized_forward(spec)
    served, served_variables = quantize_for_serving(model, variables, spec, ids)
    dtypes = {leaf.dtype for leaf in jax.tree.leaves(served_variables["params"])}
    assert jnp.dtype(jnp.int8) in dtypes
    logits = served.apply(served_variables, ids)
    np.testing.assert_array_equal(logits, qmodel.apply(variables, ids))
    assert not np.array_equal(logits, model.apply(variables, ids))


@pytest.mark.parametrize("dtype", ["int8", "fp8"])
@pytest.mark.parametrize("weight_only", [False, True])
def test_a_quantized_text_task_matches_qwix_direct_logits_and_generation(dtype, weight_only):
    """Qwix 0.1.8 PTQ with fp32 compute, compared bitwise with the public task.

    Greedy agreement with bf16 compute over the same fp32 master weights is
    reported separately; quantization changes the policy.
    """
    import functools

    from dew.inference import TextGeneration
    from dew.quantization import entry_methods
    from dew.sampling import Sampling

    qwix = pytest.importorskip("qwix")
    model = tiny()
    prompt = jnp.asarray(token_batch()["text"][:, :4])
    variables = model.init(jax.random.key(0), prompt)
    task = TextGeneration(model, variables, sampling=Sampling(temperature=0), max_new_tokens=4)
    spec = Quantization(dtype=dtype, weight_only=weight_only, patterns=(".*_proj",))
    served = task.quantized(spec)
    qtype = jnp.int8 if dtype == "int8" else jnp.float8_e4m3fn
    fields = {"op_names": ("dot_general", "einsum", "dot")} if weight_only else {}
    rules = [qwix.QuantizationRule(module_path=".*_proj", weight_qtype=qtype,
                                  act_qtype=None if weight_only else qtype, **fields)]
    reference = quantize_with(model, qwix.PtqProvider(rules),
                                    methods=entry_methods(model))
    abstract = jax.eval_shape(functools.partial(reference.init, jax.random.key(0),
                                               jnp.zeros((1, 1), jnp.int32)))
    expected_variables = {
        **variables,
        "params": qwix.quantize_params(variables["params"], abstract["params"]),
    }
    expected_logits = jax.jit(reference.apply)(expected_variables, prompt)
    logits = jax.jit(served.model.apply)(served.variables, prompt)
    np.testing.assert_array_equal(logits, expected_logits)
    assert not np.array_equal(logits, model.apply(variables, prompt))
    assert jnp.dtype(qtype) in {leaf.dtype for leaf in jax.tree.leaves(served.variables)}
    assert jnp.dtype(qtype) not in {leaf.dtype for leaf in jax.tree.leaves(task.variables)}
    actual = served(prompt, key=7).host()
    expected = TextGeneration(reference, expected_variables, sampling=task.sampling,
                              max_new_tokens=task.max_new_tokens)(prompt, key=7).host()
    for found, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(found, wanted)
    bf16 = TextGeneration(model.clone(dtype=jnp.bfloat16), variables,
                           sampling=task.sampling, max_new_tokens=task.max_new_tokens)(prompt, key=7).host()
    matches = actual.tokens[:, -4:] == bf16.tokens[:, -4:]
    print(f"{jax.default_backend()} {dtype} weight_only={weight_only}: Qwix logit error=0; "
          f"bf16 greedy token agreement={int(matches.sum())}/{matches.size}; "
          f"whole continuations={int(matches.all(axis=1).sum())}/{matches.shape[0]}")


FLUX_BLOCKS = (r".*transformer_blocks_\d+/(attn/(to_|add_).+|ff.*)",)
"""The FLUX blocks' projections and feed-forwards, at `flux_task`'s 64 features."""


def flux_source(root: Path):
    """The tiny FLUX source's text-to-image task, unpacked under `root`: a
    matmul-only denoiser beside CLIP and T5 towers and a VAE. Its denoiser
    is widened to 64 features over freshly drawn weights: the source's 24
    make int8 GEMMs cuBLAS 13 refuses on an A100, as do its 16-wide inputs
    and outputs, which `FLUX_BLOCKS` leaves in float."""
    import tarfile

    from dew.interop.pretrained import Pretrained

    with tarfile.open(Path(__file__).parent / "fixtures/flux_source.tar.xz") as archive:
        archive.extractall(root, filter="data")
    task = Pretrained.load(root / "pipeline", dtype="float32", attention_impl="xla").text_to_image()
    model = task.model.clone(heads=2, head_dim=32, axes_dims_rope=(8, 12, 12))
    example = task.prepare("", key=0, steps=1)
    params = model.init(jax.random.key(0), example.noise, jnp.zeros(example.noise.shape[:1]),
                        **example.conditions)["params"]
    return dataclasses.replace(task, model=model, variables={**task.variables, "params": params})


@pytest.fixture(scope="module")
def flux_task(tmp_path_factory):
    return flux_source(tmp_path_factory.mktemp("flux"))


@pytest.mark.parametrize("dtype", ["int8", "fp8"])
@pytest.mark.parametrize("weight_only", [False, True])
def test_a_quantized_image_task_matches_qwix_direct_prediction_and_samples(flux_task, dtype, weight_only):
    """Qwix 0.1.8 PTQ over the task's denoiser, built here from Qwix's own
    rules, provider and `quantize_params`, is the reference for
    `TextToImage.quantized`: the same quantized kernels, the denoiser's
    prediction bitwise, and a decoded two-step sample bitwise. The towers
    and the VAE keep their weights, leaf for leaf."""
    import functools

    from dew.quantization import entry_methods

    qwix = pytest.importorskip("qwix")
    task = flux_task
    served = task.quantized(Quantization(dtype=dtype, weight_only=weight_only, patterns=FLUX_BLOCKS))
    qtype = jnp.int8 if dtype == "int8" else jnp.float8_e4m3fn
    fields = {"op_names": ("dot_general", "einsum", "dot")} if weight_only else {}
    rules = [qwix.QuantizationRule(module_path=FLUX_BLOCKS[0], weight_qtype=qtype,
                                  act_qtype=None if weight_only else qtype, **fields)]
    reference = quantize_with(task.model, qwix.PtqProvider(rules),
                                    methods=entry_methods(task.model))
    example = task.prepare("", key=0, steps=1)
    own = {name: tree for name, tree in task.variables.items() if name not in ("encoders", "autoencoder")}
    abstract = jax.eval_shape(functools.partial(reference.init, jax.random.key(0), example.noise,
                                                jnp.zeros(example.noise.shape[:1]), **example.conditions))
    expected = {**own, "params": qwix.quantize_params(jax.tree.map(jnp.asarray, own["params"]),
                                                      abstract["params"])}
    def assert_same_leaves(found, wanted):
        for leaf, other in zip(jax.tree.leaves(found), jax.tree.leaves(wanted), strict=True):
            np.testing.assert_array_equal(leaf, other)

    assert_same_leaves(served.variables["params"], expected["params"])
    for frozen in ("encoders", "autoencoder"):
        assert_same_leaves(served.variables[frozen], task.variables[frozen])

    latent = jax.random.normal(jax.random.key(3), example.noise.shape)
    times = jnp.full(example.noise.shape[:1], 0.6)
    served_denoiser = {name: tree for name, tree in served.variables.items() if name in own}
    prediction = jax.jit(served.model.apply)(served_denoiser, latent, times, **example.conditions)
    wanted = jax.jit(reference.apply)(expected, latent, times, **example.conditions)
    np.testing.assert_array_equal(prediction, wanted)
    unquantized = task.model.apply(own, latent, times, **example.conditions)
    assert not np.array_equal(prediction, unquantized)
    direct = dataclasses.replace(task, model=reference, variables={**task.variables, **expected})
    np.testing.assert_array_equal(served(["a red bird"], steps=2, key=1).host().images,
                                  direct(["a red bird"], steps=2, key=1).host().images)


def test_quantizing_host_weights_keeps_numpy_storage_and_matches_qwix():
    """Host kernels use Qwix 0.1.8 on the active backend and retain host storage."""
    import functools

    from dew.inference import TextGeneration
    from dew.quantization import entry_methods
    from dew.sampling import Sampling

    qwix = pytest.importorskip("qwix")
    model = tiny()
    prompt = jnp.asarray(token_batch()["text"][:, :4])
    resident = model.init(jax.random.key(0), prompt)
    host = jax.device_get(resident)
    spec = Quantization(weight_only=True, patterns=(".*_proj",))
    task = TextGeneration(model, host, sampling=Sampling(temperature=0))
    served = task.quantized(spec)
    assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(served.variables))
    # The weights its generations read are placed once, a quantized kernel's buffers included.
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(served.placed))
    reference = quantize_with(model, qwix.PtqProvider([
        qwix.QuantizationRule(module_path=".*_proj", weight_qtype=jnp.int8,
                              op_names=("dot_general", "einsum", "dot"))]),
        methods=entry_methods(model))
    abstract = jax.eval_shape(functools.partial(reference.init, jax.random.key(0),
                                               jnp.zeros((1, 1), jnp.int32)))
    reference_variables = {**host, "params": qwix.quantize_params(resident["params"], abstract["params"])}
    expected = TextGeneration(reference, jax.device_get(reference_variables), sampling=task.sampling)
    np.testing.assert_array_equal(jax.jit(served.model.apply)(served.variables, prompt),
                                  jax.jit(expected.model.apply)(expected.variables, prompt))
    np.testing.assert_array_equal(served(prompt, 3, key=7).host().tokens,
                                  expected(prompt, 3, key=7).host().tokens)


PACKED_SEEDS = 256
"""The packed quantized pipeline's declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for`
on armada."""


def test_quantizing_packed_pipeline_weights_matches_the_canonical_quantized_model():
    """`dew.pipeline` packs a decoder's projections before quantizing them; served packed, its logits
    and its draws' log-likelihoods, a task's and a server's, are on par (check 1) with the canonical
    tree's quantized the same way, over drawn prompts, from the quantized model's own float64 value:
    the unquantized model in float64 over the canonical served weights dequantized exactly (int8
    times scale in float64; weight-only, the activations are not quantized). The greedy draws are the
    canonical ones, and their behaviour log-likelihoods its exact zeros."""
    from pathlib import Path

    import dew
    from dew.inference.serving import Server
    from dew.interop import Pretrained
    from dew.quantization import Annotated
    from dew.sampling import Sampling

    qwix = pytest.importorskip("qwix")
    directory = Path(__file__).parent / "fixtures/hf/llama-tiny"
    spec = Quantization(weight_only=True, patterns=(".*_proj",))
    canonical = Pretrained.load(directory, dtype="float32").text_generation(
        sampling=Sampling(temperature=0, eos_token_ids=None))
    packed = dew.pipeline(str(directory), dtype="float32")
    reference = canonical.quantized(spec)
    served = packed.quantized(spec)
    server = Server.from_task(served, slots=max(2, jax.device_count()), capacity=64)
    vocab = canonical.model.vocab_size

    def prompt_of(seed) -> np.ndarray:
        return np.random.default_rng(int(seed)).integers(1, vocab, size=(1, 3)).astype(np.int32)

    @functools.cache
    def drawn(seed: int):
        """The canonical task's, the packed task's and the packed server's draws from the seed's prompt."""
        prompt = prompt_of(seed)
        expected = reference(prompt, 4, key=seed).host()
        actual = served(prompt, 4, key=seed).host()
        from_server = server([prompt[0].tolist()], 4, key=seed)[0].host()
        for found in (actual, from_server):
            np.testing.assert_array_equal(found.tokens, expected.tokens)
            np.testing.assert_array_equal(found.behavior_log_probs, expected.behavior_log_probs)
        return expected, actual, from_server

    def raw(generation):
        return generation.raw_log_probs[0, :generation.lengths[0]]

    @as_given
    def packed_side(seed):
        _, actual, from_server = drawn(int(seed))
        return {"logits": np.asarray(served.model.apply(served.variables, prompt_of(seed))),
                "task": raw(actual), "server": raw(from_server)}

    @as_given
    def canonical_side(seed):
        expected, _, _ = drawn(int(seed))
        return {"logits": np.asarray(reference.model.apply(reference.variables, prompt_of(seed))),
                "task": raw(expected), "server": raw(expected)}

    def exact(params):
        """The served weights as the float64 values they hold: each quantized kernel's integers times
        its scales, computed in float64."""
        def value(node):
            return qwix.dequantize(node.array.astype(jnp.float64)) if isinstance(node, Annotated) else node
        return jax.tree.map(value, params, is_leaf=lambda node: isinstance(node, Annotated))

    wide = canonical.model.clone(dtype=jnp.float64)

    def truth(seed):
        """The quantized model's float64 logits over the seed's prompt and its raw log-likelihood of
        the canonical draw's tokens, teacher-forced."""
        expected, _, _ = drawn(int(seed))
        weights = jax.tree.map(lambda leaf: np.asarray(leaf, np.float64)
                               if np.issubdtype(np.asarray(leaf).dtype, np.floating) else leaf,
                               {**reference.variables, "params": exact(reference.variables["params"])})
        prompt = prompt_of(seed)
        tokens = expected.tokens[0, expected.prompt_width:expected.prompt_width + expected.lengths[0]]
        log_probs = jax.nn.log_softmax(wide.apply(weights, np.concatenate([prompt[0], tokens])[None])[0], -1)
        width = prompt.shape[1]
        picked = np.asarray(log_probs[np.arange(width - 1, width - 1 + len(tokens)), tokens])
        return {"logits": np.asarray(wide.apply(weights, prompt)), "task": picked, "server": picked}

    assert_as_exact(packed_side, canonical_side, lambda seed: (np.int32(seed),), "packed quantized pipeline",
                    seeds=PACKED_SEEDS, truth_fn=truth)


def test_a_program_finds_every_quantized_tower_its_models_hold():
    """The options a program compiles with read the quantization of every model it is handed and every
    tower those hold (`quantizations`): a quantized text transformer inside a condition's encoder record,
    inside the tuple of a task's encoders, counts as a quantized denoiser handed alone does, and an
    unquantized one adds nothing."""
    from dew.nn.kernels.options import quantizations

    pytest.importorskip("qwix")

    @dataclasses.dataclass(frozen=True)
    class Encoder:
        transformer: nn.Module
        params: dict

    spec = Quantization(dtype="int8")
    plain = tiny()
    quantized = spec.apply(plain)
    weights = {"params": {"kernel": np.zeros((2, 2), np.float32)}}
    assert quantizations(plain, (("textcontext", Encoder(plain, weights)),)) == []
    held = quantizations((("textcontext", Encoder(quantized, weights)),))
    assert [found.dtype for found in held] == ["int8"]
    assert quantizations(quantized) == held
    # A FrozenDict makes each nested mapping as it is read: a tower behind fifty such siblings is found.
    nested = FrozenDict({f"branch_{index}": {"leaf": {"value": np.zeros(())}} for index in range(50)}
                        | {"tower": {"encoder": Encoder(quantized, weights)}})
    assert [found.dtype for found in quantizations(nested)] == ["int8"]


def test_a_diffusion_step_compiles_for_the_quantized_text_tower_it_encodes_with():
    """A text-to-image objective's programs run its conditions' encoders and its autoencoder beside its
    denoiser (`Objective.towers`), so the options its step and its samples compile with read a quantized
    text tower (`compiler_options`) as they read a quantized denoiser."""
    from pathlib import Path

    from dew.diffusion import presets
    from dew.inputs import Condition, Field, InputSpec
    from dew.inputs.encoders import CLIPText
    from dew.nn.backbones.dit import SimpleDiT
    from dew.nn.kernels.options import quantizations
    from dew.objectives.diffusion import DiffusionObjective

    pytest.importorskip("qwix")
    encoder = CLIPText.from_pretrained(str(Path(__file__).parent / "fixtures" / "clip" / "tiny"))

    def objective(tower):
        return DiffusionObjective(SimpleDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2,
                                            mlp_ratio=1),
                                  presets.Flow(), InputSpec(Field("image", (8, 8, 3)),
                                                            {"textcontext": Condition(tower)}),
                                  guidance=None, steps=2)

    assert quantizations(*objective(encoder).towers()) == []
    tower = Quantization(dtype="int8").apply(encoder.transformer)
    quantized = dataclasses.replace(encoder, transformer=tower)
    assert [found.dtype for found in quantizations(*objective(quantized).towers())] == ["int8"]


SPLIT_SEEDS = 16
"""The split quantized tree's declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for` on
armada."""


def test_an_objective_over_a_quantized_packed_task_holds_the_canonical_tree():
    """A task's tree quantized for serving holds its projections packed; an objective built over it, a
    distillation teacher's, holds the model's canonical tree (`dew.inference.projections.unpacked`), each
    member the packed kernel's columns quantized as they were (`split_columns`), and computes logits on
    par (check 1) with the served task's over drawn prompts, from the quantized model's own float64
    value: the unquantized model in float64 over the served packed weights dequantized exactly. A
    rule that quantizes activations and names the packed site alone keeps that group packed, its members
    named by no rule: the objective's forward is the served task's."""
    from pathlib import Path

    import dew
    from dew.interop import Pretrained
    from dew.objectives.lm import LMObjective
    from dew.quantization import Annotated

    qwix = pytest.importorskip("qwix")
    directory = Path(__file__).parent / "fixtures/hf/llama-tiny"
    served = dew.pipeline(str(directory), dtype="float32").quantized(
        Quantization(weight_only=True, patterns=(".*_proj",)))
    assert "qkv_proj" in served.variables["params"]["layers_0"]["self_attn"]
    held = LMObjective(served.model, 3, variables=served.variables).variables
    layer, packed_layer = held["params"]["layers_0"], served.variables["params"]["layers_0"]
    assert "qkv_proj" not in layer["self_attn"] and "gate_up_proj" not in layer["mlp"]
    for (block, packed_name), names in ((("self_attn", "qkv_proj"), ("q_proj", "k_proj", "v_proj")),
                                        (("mlp", "gate_up_proj"), ("gate_proj", "up_proj"))):
        if packed_name not in packed_layer[block]:
            continue
        members = [qwix.dequantize(layer[block][name]["kernel"].array) for name in names]
        np.testing.assert_array_equal(np.concatenate(members, axis=-1),
                                      qwix.dequantize(packed_layer[block][packed_name]["kernel"].array))
    canonical = Pretrained.load(directory, dtype="float32").model
    wide = canonical.clone(dtype=jnp.float64)
    vocab = canonical.vocab_size

    def draw(seed):
        return (np.random.default_rng([0x1313, seed]).integers(1, vocab, size=(2, 5)).astype(np.int32),)

    def value(node):
        """A leaf in float64: a quantized kernel its integers times its scales."""
        if isinstance(node, Annotated):
            return np.asarray(qwix.dequantize(node.array.astype(jnp.float64)), np.float64)
        floating = np.issubdtype(np.asarray(node).dtype, np.floating)
        return np.asarray(node, np.float64) if floating else np.asarray(node)

    with jax.enable_x64(new_val=True):
        weights = jax.tree.map(value, dict(served.variables),
                               is_leaf=lambda node: isinstance(node, Annotated))

    assert_as_exact(lambda tokens: served.model.apply(held, tokens),
                    lambda tokens: served.model.apply(served.variables, tokens), draw,
                    "split quantized tree", seeds=SPLIT_SEEDS,
                    truth_fn=lambda tokens: wide.apply(weights, tokens))
    packed_only = dew.pipeline(str(directory), dtype="float32").quantized(
        Quantization(patterns=(".*/qkv_proj",)))
    kept = LMObjective(packed_only.model, 3, variables=packed_only.variables).variables
    assert "qkv_proj" in kept["params"]["layers_0"]["self_attn"]
    tokens = draw(0)[0]
    np.testing.assert_array_equal(packed_only.model.apply(kept, tokens),
                                  packed_only.model.apply(packed_only.variables, tokens))


def test_a_quantized_multimodal_task_keeps_its_processor_and_media():
    from pathlib import Path

    from dew.interop import Pretrained
    from dew.sampling import Sampling

    pytest.importorskip("qwix")
    directory = Path(__file__).parent / "fixtures/hf/gemma3-native-tiny"
    source = Pretrained.load(directory, dtype="float32", attention_impl="reference", max_seq_len=64)
    task = source.text_generation(sampling=Sampling(temperature=0))
    image = np.load(directory / "raw_images.npy")[0]
    prompt = "token7 <start_of_image> token9"
    inputs = source.processor(prompt, images=[image])
    spec = Quantization(weight_only=True, patterns=(".*_proj",))
    served = task.quantized(spec, example=inputs)
    model, variables = quantize_for_serving(
        task.model, task.variables, spec, inputs.tokens, **inputs.kwargs()
    )
    np.testing.assert_array_equal(served.model.apply(served.variables, inputs.tokens, **inputs.kwargs()),
                                  model.apply(variables, inputs.tokens, **inputs.kwargs()))
    generated = served(prompt, 3, key=0, images=[image])
    assert generated.text == task.decode(generated)
    assert len(generated.text) == 1 and generated.text[0]
    assert served.processor is task.processor and served.sampling == task.sampling


def test_a_bf16_int8_matmul_scales_its_int32_products_in_float32():
    """A served bf16 Dense multiplies its int8 matmul's int32 accumulator by
    both scales in float32 and rounds once to bf16. Qwix 0.1.8 rounded the
    accumulator to bf16 before the scales multiplied in, and XLA:TPU then
    compiled the int8 matmuls of the bf16 176M text-to-image model with bf16
    results, which sampled NaN images on a v6e."""
    pytest.importorskip("qwix")
    from flax import linen as nn
    from qwix._src.core import qarray

    dense = nn.Dense(64, use_bias=False, dtype=jnp.bfloat16)
    x = jax.random.normal(jax.random.key(0), (8, 256), jnp.bfloat16)
    variables = dense.init(jax.random.key(1), x)
    served, served_variables = quantize_for_serving(dense, variables, Quantization(), x)
    kernel = served_variables["params"]["kernel"].array.astype(jnp.bfloat16)
    activations = qarray.quantize(x, qarray.HowToQuantize(qtype=jnp.int8, channelwise_axes=(0,)))
    products = jax.lax.dot(activations.qvalue, kernel.qvalue, preferred_element_type=jnp.int32)
    scaled = products * activations.scale.astype(jnp.float32) * kernel.scale.astype(jnp.float32)
    np.testing.assert_array_equal(served.apply(served_variables, x), scaled.astype(jnp.bfloat16))


def test_quantized_training_differentiates_through_complex_matmuls_in_float():
    """Quantized training leaves a matmul with a complex operand, like the
    S5 scan's in the hybrid DiT, in float, and differentiates through it:
    values and gradients are those of Qwix's own quantized training applied
    to the real Dense alone. Observed on CPU: bitwise equal, 2.1e-02 from fp32
    in the kernel's gradient. Before, Qwix quantized the real operand of a
    real-by-complex matmul, its backward returned a complex gradient for
    that real operand, and `jax.grad` failed on it. The kernel's gradient is
    not the float one: quantization reached the Dense."""
    qwix = pytest.importorskip("qwix")
    from flax import linen as nn

    class Rotated(nn.Module):
        @nn.compact
        def __call__(self, x):
            # A real activation times a complex matrix, as the S5 scan's
            # input projection is, then two complex operands.
            rotation = jnp.exp(1j * jnp.arange(64.0).reshape(8, 8))
            mixed = jnp.einsum("bf,fg->bg", nn.Dense(8, name="proj")(x), rotation)
            return jnp.real(jax.lax.dot_general(mixed, rotation, (((1,), (0,)), ((), ()))))

    model = Rotated()
    x = jax.random.normal(jax.random.key(0), (4, 16))
    variables = model.init(jax.random.key(1), x)
    reference = quantize_with(model, qwix.QtProvider(
        [qwix.QtRule(module_path="proj", weight_qtype=jnp.int8, act_qtype=jnp.int8)]))

    def value_and_gradients(module):
        return jax.jit(jax.value_and_grad(lambda v, x: jnp.sum(module.apply(v, x) ** 2), argnums=(0, 1)))(
            variables, x)

    got = value_and_gradients(Quantization().apply(model))
    want = value_and_gradients(reference)
    jax.tree.map(np.testing.assert_array_equal, got, want)
    _, (kernel_gradient, _) = value_and_gradients(model)
    assert not np.array_equal(got[1][0]["params"]["proj"]["kernel"],
                              kernel_gradient["params"]["proj"]["kernel"])


def test_serving_leaves_complex_matmuls_in_float():
    """Qwix quantizes real values only; a complex matmul, like the S5 scan's
    in the hybrid DiT, runs unquantized beside the quantized Dense: the
    served forward is Qwix's own serving of the Dense alone, bit for bit, and
    not the float forward. Observed on CPU: 4e-03 relative from float."""
    import functools

    from dew.quantization import entry_methods

    qwix = pytest.importorskip("qwix")
    from flax import linen as nn

    class Rotated(nn.Module):
        @nn.compact
        def __call__(self, x):
            h = nn.Dense(8)(x).astype(jnp.complex64)
            rotation = jnp.exp(1j * jnp.arange(64.0).reshape(8, 8))
            mixed = jnp.einsum("bf,fg->bg", h, rotation)
            return jnp.real(jax.lax.dot_general(mixed, rotation, (((1,), (0,)), ((), ()))))

    model = Rotated()
    x = jax.random.normal(jax.random.key(0), (4, 16))
    variables = model.init(jax.random.key(1), x)
    served, served_variables = quantize_for_serving(model, variables, Quantization(), x)
    reference = quantize_with(model, qwix.PtqProvider([qwix.QuantizationRule(
        module_path="Dense_0", weight_qtype=jnp.int8, act_qtype=jnp.int8)]), methods=entry_methods(model))
    abstract = jax.eval_shape(functools.partial(reference.init, jax.random.key(1), x))
    expected = {**variables, "params": qwix.quantize_params(variables["params"], abstract["params"])}
    found = served.apply(served_variables, x)
    np.testing.assert_array_equal(found, reference.apply(expected, x))
    assert not np.array_equal(found, model.apply(variables, x))


PIPELINE_SEEDS = 512
"""The quantized pipeline's declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for` on
armada."""


@pytest.mark.mesh
def test_a_quantized_pipeline_has_finite_loss_and_gradients():
    """The int8 trunk over two stages on the eight simulated devices: finite
    loss and gradients, and the loss on par with the plain wrapped loop's
    (check 1) from the unquantized model's float64 loss, over drawn weights
    and tokens: the pipeline quantizes its microbatches as the loop
    quantizes the batch, the way the bf16 pipeline test holds its policy's
    rounding. (The gradients are not drawn: int8 rounding is
    discontinuous, so a gradient's error from the float model jumps where a
    value crosses a level, and the loop's own does not resolve within 512
    seeds.)"""
    pytest.importorskip("qwix")
    model = tiny(num_layers=4)
    qmodel = Quantization().apply(model)
    info = Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(3), ema=None)
    init = jax.jit(model.init)

    def draw(seed):
        variables = init(jax.random.key(seed), jnp.ones((1, SEQ_LEN), jnp.int32))
        rows = np.random.default_rng(seed).integers(0, VOCAB, size=(8, SEQ_LEN + 1)).astype(np.int32)
        # A step's loss, taken with its backward under its own cotangent, one.
        return variables, rows, np.float32(1)

    def gradient(objective):
        def step(params, variables, batch, cotangent):
            return with_gradients(lambda params: objective.scalar_loss(
                {**variables, "params": params}, batch, info)[0])(params, cotangent)
        return jax.jit(step)

    def runner(spec, microbatches=None):
        mesh = spec.build()
        program = gradient(LMObjective(qmodel, SEQ_LEN))

        @as_given
        def run(variables, rows, cotangent):
            placed = jax.device_put(variables, Layout(min_shard=256).shardings(mesh, variables))
            with jax.set_mesh(mesh), pipeline_microbatches(microbatches):
                value, grads = program(placed["params"], placed, shard_batch(mesh, {"text": rows}), cotangent)
            return np.asarray(value), jax.tree.map(np.asarray, grads)
        return run

    unquantized = gradient(LMObjective(model.clone(dtype=jnp.float64), SEQ_LEN))
    loop, piped = runner(MeshSpec(fsdp=8)), runner(MeshSpec(fsdp=4, stage=2), 2)
    loss, grads = piped(*draw(0))
    assert np.isfinite(loss) and all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(grads))
    assert_as_exact(as_given(lambda *inputs: {"loss": piped(*inputs)[0]}),
                    as_given(lambda *inputs: {"loss": loop(*inputs)[0]}),
                    draw, "quantized pipeline loss", seeds=PIPELINE_SEEDS,
                    truth_fn=lambda variables, rows, cotangent: {"loss": np.asarray(unquantized(
                        variables["params"], variables, {"text": rows}, cotangent)[0])})


def test_stochastic_rounding_draws_from_its_own_stream():
    """Test rounding without the decoder embedding scatter's GPU reduction order."""
    pytest.importorskip("qwix")
    from flax import linen as nn

    model = nn.Dense(16)
    values = jax.random.normal(jax.random.key(3), (32, 8))
    variables = model.init(jax.random.key(2), values)
    quantized = Quantization(bwd_qtype="int8", bwd_stochastic_rounding="uniform").apply(model)

    def loss(variables, key):
        result = quantized.apply({"params": variables}, values,
                                 rngs={"stochastic_rounding": key})
        return jnp.mean(result ** 2)

    differentiate = jax.jit(jax.grad(loss))
    gradients = differentiate(variables["params"], jax.random.key(0))
    repeated = differentiate(variables["params"], jax.random.key(0))
    other = differentiate(variables["params"], jax.random.key(1))
    for first, second in zip(jax.tree.leaves(gradients), jax.tree.leaves(repeated), strict=True):
        np.testing.assert_array_equal(first, second)
    assert max(float(jnp.max(jnp.abs(first - second)))
               for first, second in zip(jax.tree.leaves(gradients),
                                        jax.tree.leaves(other), strict=True)) > 0.0


def test_an_objective_rounds_its_gradients_from_the_step_key():
    """An LM objective's training loss under int8 gradients with stochastic
    rounding: the step's key alone feeds Qwix's rounding stream, which the
    objective folds off the key it hands dropout. The same key repeats the
    gradients, as every process of a pool holds the same step key; another
    key moves them, and deterministic rounding does not move with the key.
    (The cuda lane runs the suite under --xla_gpu_deterministic_ops, so a
    repeat is bitwise there too.)"""
    pytest.importorskip("qwix")
    batch = token_batch()

    def gradients(spec, key):
        _, qmodel, variables, _ = quantized_forward(spec)
        objective = LMObjective(qmodel, SEQ_LEN)
        step = Step(step=jnp.zeros((), jnp.int32), key=key, ema=None)
        return jax.jit(jax.grad(lambda params: objective.scalar_loss(
            {**variables, "params": params}, batch, step), has_aux=True))(variables["params"])[0]

    stochastic = Quantization(bwd_qtype="int8", bwd_stochastic_rounding="uniform")
    drawn = gradients(stochastic, jax.random.key(0))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(drawn))
    jax.tree.map(np.testing.assert_array_equal, drawn, gradients(stochastic, jax.random.key(0)))
    assert not all(np.array_equal(first, second) for first, second in zip(
        jax.tree.leaves(drawn), jax.tree.leaves(gradients(stochastic, jax.random.key(1))), strict=True))
    fixed = Quantization(bwd_qtype="int8")
    jax.tree.map(np.testing.assert_array_equal, gradients(fixed, jax.random.key(0)),
                 gradients(fixed, jax.random.key(1)))


# --------------------------------------------------------------------------
# The trainer's knob
# --------------------------------------------------------------------------

RES = 8
RUN_BATCH = 8
"""A whole-run batch: one record per simulated device, so the default mesh
splits it evenly."""


def image_batches(batch):
    def stream():
        rng = np.random.RandomState(0)
        while True:
            yield {"image": rng.randint(0, 256, (batch, RES, RES, 3), np.uint8)}
    return stream


def diffusion_run(directory, batch=RUN_BATCH, **trainer):
    """The smallest unconditional diffusion run: a tiny DiT over 8-pixel
    images, one step, nothing written but the record."""
    from diffusion_support import images_of

    from dew.config import TrainerConfig
    from dew.config.diffusion import DiffusionRunConfig
    from dew.registry import ModelConfig

    return DiffusionRunConfig(
        model=ModelConfig("simple_dit", {"patch_size": 4, "emb_features": 16,
                                         "num_layers": 1, "num_heads": 2, "dtype": "float32"}),
        data=images_of("tfds://oxford_flowers102", RES, batch=batch), text=None,
            val_metrics=(),
        trainer=TrainerConfig(name="quantized", checkpoint_dir=str(directory),
                              steps=1, eval_every=None, checkpoint_every=None,
                              compilation_cache_dir=None, **trainer), objective=ObjectiveConfig("diffusion",
                              {"guidance": None, "steps": 2}))


def test_the_trainer_knob_quantizes_the_objective_a_run_trains(tmp_path):
    """`--trainer.quantization` is the one place a run names quantization:
    `RunConfig.train` wraps the module the objective trains before anything
    initialises it, and the run steps on the quantized forward. The distance
    from the fp32 forward on the trained weights is the assertion that the
    rules reached the matmuls. Observed on CPU: 2.7e-05 on activations of
    order 1e-2, against 0.0 for a run trained without the knob."""
    pytest.importorskip("qwix")
    from dew.data.dataset import Loader

    batch = RUN_BATCH
    config = diffusion_run(tmp_path, batch, quantization=Quantization())
    objective = config.build()
    plain = config.build()

    state = config.train(objective, Loader(lambda partition: image_batches(batch)(), None, None, batch),
                         name="quantized")

    assert int(state.step) == 1
    image = jnp.ones((1, RES, RES, 3), jnp.float32)
    noise_level = jnp.ones((1,), jnp.float32)
    # Compiled, as a run computes: XLA:GPU compiles an int8 convolution only
    # with its dequantization fused in, and eagerly the convolution runs alone.
    quantized_out = jax.jit(objective.model.apply)(state.variables, image, noise_level)
    plain_out = jax.jit(plain.model.apply)(state.variables, image, noise_level)
    assert float(jnp.max(jnp.abs(quantized_out - plain_out))) > 0.0


def test_the_trainer_knob_quantizes_what_a_distillation_trains_and_not_its_teacher():
    """The knob wraps the modules an objective trains (`ProgramModule.trained`):
    a distillation's student computes quantized, and its frozen teacher
    keeps its own numerics bit for bit."""
    pytest.importorskip("qwix")
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.distillation import DistillationObjective
    from dew.objectives.lm import LMObjective
    from dew.training.quantization import quantize_trunk

    # Every int8 GEMM 32 wide or more: cuBLAS 13 refuses 16-wide ones on an A100.
    def decoder():
        return CausalTransformer(vocab_size=32, emb_features=64, num_layers=1, num_heads=2, mlp_features=128,
                                 max_seq_len=32)

    student, teacher = decoder(), decoder()
    objective = DistillationObjective(LMObjective(student, seq_len=4), LMObjective(teacher, seq_len=4))
    quantize_trunk(objective, Quantization())
    tokens = jnp.arange(32, dtype=jnp.int32)[None]
    variables = student.init(jax.random.key(0), tokens)
    trained, frozen = (entry.module for entry in objective.program_key())
    assert float(jnp.max(jnp.abs(jax.jit(trained.apply)(variables, tokens)
                                 - jax.jit(student.apply)(variables, tokens)))) > 0.0
    np.testing.assert_array_equal(jax.jit(frozen.apply)(variables, tokens),
                                  jax.jit(teacher.apply)(variables, tokens))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("precision", [None, "highest"])
def test_a_matmul_quantizes_whatever_precision_it_pins(precision, dtype):
    """A quantized product has no precision to honour, so a matmul a rule matches quantizes
    whatever precision it pins: a float32 model's layers pin float32 products
    (`float32_products`) and quantize as a bfloat16 model's do."""
    x = jnp.linspace(-2, 2, 16, dtype=jnp.float32).reshape(2, 8).astype(dtype)
    model = nn.Dense(16, dtype=dtype, param_dtype=dtype, precision=precision)
    variables = model.init(jax.random.key(0), x)
    quantized = np.asarray(Quantization().apply(model).apply(variables, x), np.float32)
    assert not np.array_equal(quantized, np.asarray(model.apply(variables, x), np.float32))


class Calls(nn.Module):
    """A model whose call is `function` of its operands, for `Quantization.apply` to wrap."""

    function: Callable[..., jax.Array]

    def __call__(self, *operands):
        return self.function(*operands)


def zone_cases(dtype) -> dict[str, tuple[Callable[..., jax.Array], tuple, dict]]:
    """Each `UNQUANTIZED` function, its operands in `dtype` and its keywords."""
    rng = np.random.default_rng(0)

    def normal(*shape):
        return jnp.asarray(rng.standard_normal(shape), jnp.float32).astype(dtype)

    pole = jnp.exp(jax.lax.complex(-jnp.linspace(0.05, 0.4, 4), jnp.linspace(0.3, 2.0, 4)))
    # Each row of -1, 0 and 1 sits one step from one codeword and a step and 1/128 across from
    # another: a gap of 2**-14 in squared distance, which the float scores, sums of multiples of
    # 2**-14 below 2**7, hold exactly in either dtype, and an int8 product's rounding reorders.
    whole = np.random.default_rng(1).integers(-1, 2, (64, 32)).astype(np.float32)
    whole[:, -1] = 0
    steps = np.eye(32, dtype=np.float32)[np.random.default_rng(2).integers(0, 31, 64)]
    across = np.eye(32, dtype=np.float32)[-1] / 128
    rows = jnp.asarray(whole, dtype)
    codebook = jnp.asarray(np.concatenate([whole + steps, whole - steps + across]), dtype)
    return {
        "dew.nn.ssm.diagonal_recurrence": (diagonal_recurrence, (pole, normal(2, 12, 8)), {"chunk": 4}),
        "dew.nn.attention_residuals.depth_mixed": (
            depth_mixed, (normal(2, 3, 4, 8), normal(8), normal(2, 3, 4, 8)), {}),
        "dew.nn.quantizers.nearest_codewords": (nearest_codewords, (rows, codebook), {}),
        "dew.nn.kv_cache.rotated": (rotated, (normal(2, 4, 8), hadamard(8)), {}),
        "dew.lora._updated": (lora_update, (normal(8, 6), normal(8, 2), normal(2, 6)), {"scale": 2.0}),
    }


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("name", sorted(UNQUANTIZED))
def test_a_function_the_model_keeps_in_float_runs_unquantized(name, dtype):
    """Each function `UNQUANTIZED` names runs in a quantized model as in the unquantized one,
    bitwise, in a float32 and a bfloat16 model, the unquantized one summing its products as Qwix's
    do (`FP32_SUMS`), where a copy of its code under another name, which the table does not reach,
    quantizes its products and moves."""
    function, operands, keywords = zone_cases(dtype)[name]
    copy = types.FunctionType(function.__code__, function.__globals__, f"{function.__name__}_copy",
                              function.__defaults__, function.__closure__)
    copy.__kwdefaults__ = function.__kwdefaults__

    def ran(function, quantize: bool) -> np.ndarray:
        model = Calls(functools.partial(function, **keywords))
        # Qwix quantizes a product as it is traced, and jnp.matmul is jitted, so each run traces
        # afresh: one that reused another's cached trace would take its products, quantized or not.
        jax.clear_caches()
        if quantize:
            return np.asarray(Quantization().apply(model).apply({}, *operands))
        with jax.default_matmul_precision(FP32_SUMS):
            return np.asarray(model.apply({}, *operands))

    plain = ran(function, quantize=False)
    np.testing.assert_array_equal(ran(function, quantize=True), plain)
    np.testing.assert_array_equal(ran(copy, quantize=False), plain)
    assert not np.array_equal(ran(copy, quantize=True), plain)


def test_an_einsum_of_three_operands_runs_as_the_unquantized_model_runs_it():
    """Qwix quantizes a two-operand einsum alone and raises on any other, so the provider runs one
    of three operands, torch's bicubic resampling among them, as the unquantized model does where
    it sums as Qwix's products do (`FP32_SUMS`)."""
    x = jnp.asarray(np.random.default_rng(0).standard_normal((1, 4, 5, 3)), jnp.float32)
    model = Calls(functools.partial(torch_bicubic_resize, height=7, width=9))
    with jax.default_matmul_precision(FP32_SUMS):
        plain = model.apply({}, x)
    np.testing.assert_array_equal(Quantization().apply(model).apply({}, x), plain)


def assert_computed_as_on_the_host(run, label: str) -> None:
    """`run(dtype)` under the default device in float32, held as exact as the same run on the host's
    CPU in float32 to the host's float64 (`assert_as_exact_as_the_reference`).

    The host computes an int8 product exactly in either dtype, so a device whose int8 GEMM returns
    twice its product (XLA's Triton fusion, openxla/xla#51016) is further from float64 by orders of
    magnitude, while one that rounds its float parts in another order is as exact as the host."""
    cpu = jax.devices("cpu")[0]
    found = np.asarray(run(jnp.float32), np.float64)
    with jax.default_device(cpu):
        reference = np.asarray(run(jnp.float32), np.float64)
        with jax.enable_x64(new_val=True):
            truth = np.asarray(run(jnp.float64), np.float64)
    assert_as_exact_as_the_reference(found, reference, truth, label)


def placed(tree, dtype):
    """`tree`'s host arrays on the default device, its floating leaves in `dtype`."""
    return jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype if np.issubdtype(leaf.dtype, np.floating)
                                                 else leaf.dtype), tree)


def with_triton_gemms(flow) -> None:
    """Run `flow`, a function of this module, in a fresh process under the lane's flags without the
    two that keep XLA's Triton GEMM fusion out of a program: deterministic ops and
    `--xla_gpu_enable_triton_gemm=false` (tests/lane_environment.py's `REPEATABLE_GPU_FLAGS`). Its pinned
    autotuning then gives every GEMM a program leaves to Triton the fusion, as XLA's own autotuner
    does at small contractions, where it computed int8 dots as two or four times their product
    (openxla/xla#51016). XLA reads its flags once a process."""
    pytest.importorskip("qwix")
    here = Path(__file__).resolve().parent
    flags = [flag for flag in os.environ.get("XLA_FLAGS", "").split()
             if not flag.startswith(("--xla_gpu_deterministic_ops", "--xla_gpu_enable_triton_gemm"))]
    script = f"import {Path(__file__).stem} as tests; tests.{flow.__name__}()"
    # pytest's own path (pyproject's `pythonpath`: the sources and the repository root) and this directory.
    path = os.pathsep.join([str(here.parent / "src"), str(here.parent), str(here)])
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=1200,
                         cwd=here, env={**os.environ, "XLA_FLAGS": " ".join(flags),
                                        "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONPATH": path})
    assert run.returncode == 0, run.stderr[-4000:]


def test_a_served_int8_text_task_scores_its_draws_as_the_host_does():
    """`TextGeneration.quantized` in int8 draws greedily and scores each draw (its raw
    log-probabilities) as the same task on the host does: the device's int8 products are its
    operands' products. With Triton's GEMMs open to XLA (`with_triton_gemms`)."""
    with_triton_gemms(served_text_flow)


def served_text_flow():
    from dew.inference import TextGeneration
    from dew.sampling import Sampling

    prompt = token_batch()["text"][:, :4]
    held = jax.device_get(tiny().init(jax.random.key(0), jnp.asarray(prompt)))

    def run(dtype):
        task = TextGeneration(tiny(dtype=dtype), placed(held, dtype), sampling=Sampling(temperature=0),
                              max_new_tokens=4)
        served = task.quantized(Quantization(patterns=(".*_proj",)))
        return served(jnp.asarray(prompt), key=7).host().raw_log_probs

    assert_computed_as_on_the_host(run, "an int8 text task's log-probabilities")


def test_an_int8_image_task_walks_as_the_host_does():
    """`TextToImage.quantized` in int8 walks a prompt's latents as the same task does on the host,
    from the same quantized weights and encoded prompt. With Triton's GEMMs open to XLA
    (`with_triton_gemms`)."""
    with_triton_gemms(image_flow)


def image_flow():
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        task = flux_source(Path(root))
    served = task.quantized(Quantization(patterns=FLUX_BLOCKS))
    prepared, held = jax.device_get((served.prepare("a red bird", key=0, steps=2), served.variables))

    def run(dtype):
        walked = dataclasses.replace(served, model=served.model.clone(dtype=dtype),
                                     variables=placed(held, dtype))
        return walked(placed(prepared, dtype), steps=2, key=1, decode=False).host().latents

    assert_computed_as_on_the_host(run, "an int8 image task's latents")


def test_an_objective_encodes_through_an_int8_text_tower_as_the_host_does():
    """A program an objective compiles (`Objective.program`) compiles for every tower it runs
    (`Objective.towers`), so a prompt encoded there by an int8 CLIP tower, the denoiser beside it in float,
    is as exact as the host's encoding. With Triton's GEMMs open to XLA (`with_triton_gemms`)."""
    with_triton_gemms(int8_tower_flow)


def int8_tower_flow():
    from dew.diffusion import presets
    from dew.inputs import Condition, Field, InputSpec
    from dew.inputs.encoders import CLIPText
    from dew.nn.backbones.dit import SimpleDiT
    from dew.objectives.diffusion import DiffusionObjective

    encoder = CLIPText.from_pretrained(str(Path(__file__).parent / "fixtures" / "clip" / "tiny"))
    tower = Quantization(dtype="int8").apply(encoder.transformer)
    tokens = encoder.tokenize(["a red bird", "a blue fish"])
    held = jax.device_get(encoder.params)

    def run(dtype):
        encoding = dataclasses.replace(encoder, transformer=tower.clone(dtype=dtype))
        objective = DiffusionObjective(
            SimpleDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2, mlp_ratio=1, dtype=dtype),
            presets.Flow(), InputSpec(Field("image", (8, 8, 3)), {"textcontext": Condition(encoding)}),
            guidance=None, steps=2)
        encode = objective.program("prompt encoding", objective.encode)
        return encode({"textcontext": placed(held, dtype)}, {"textcontext": tokens})["textcontext"].hidden

    assert_computed_as_on_the_host(run, "an int8 text tower's encoding")


def test_an_int8_guide_walks_as_the_host_does():
    """A float image task autoguided by its own denoiser quantized to int8 (`Autoguidance`) walks a
    prompt's latents as the same task does on the host: the walk compiles for the guide it runs as
    for the denoiser. With Triton's GEMMs open to XLA (`with_triton_gemms`)."""
    with_triton_gemms(autoguided_image_flow)


def autoguided_image_flow():
    import tempfile

    from dew.sampling import Autoguidance

    with tempfile.TemporaryDirectory() as root:
        task = flux_source(Path(root))
    guide = task.quantized(Quantization(patterns=FLUX_BLOCKS))
    weaker = {name: tree for name, tree in guide.variables.items() if name not in ("encoders", "autoencoder")}
    prepared, held = jax.device_get((task.prepare("a red bird", key=0, steps=2),
                                     {**task.variables, "guide": weaker}))

    def run(dtype):
        walked = dataclasses.replace(task, model=task.model.clone(dtype=dtype), variables=placed(held, dtype),
                                     guidance=Autoguidance(2.0, guide.model.clone(dtype=dtype)))
        return walked(placed(prepared, dtype), steps=2, key=1, decode=False).host().latents

    assert_computed_as_on_the_host(run, "an int8 guide's latents")


def test_an_int8_decision_task_answers_as_the_host_does():
    """`Decide` over a backbone quantized to int8 gives each option the probability the same task
    gives it on the host. With Triton's GEMMs open to XLA (`with_triton_gemms`)."""
    with_triton_gemms(decision_flow)


def decision_flow():
    from test_decision_objective import INTENT, LAYOUT, SPECIALS, TOKENIZER

    from dew.decision import Decide, DecisionObjective
    from dew.decision.model import DecisionModel
    from dew.nn.backbones.causal_transformer import CausalTransformer

    backbone = CausalTransformer(vocab_size=256, emb_features=32, num_layers=2, num_heads=2, mlp_features=64,
                                 max_seq_len=128, attention_impl="xla")
    base = DecisionObjective(backbone, tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT)
    held = jax.device_get(base.init(jax.random.key(0)))
    spec = Quantization(patterns=(".*_proj",))

    def run(dtype):
        model = DecisionModel(spec.apply(backbone.clone(dtype=dtype)), base.model.head.clone(dtype=dtype))
        task = Decide(model, placed(held, dtype), LAYOUT, TOKENIZER, SPECIALS)
        return task("charged twice!", {"team": INTENT})["team"].probabilities

    assert_computed_as_on_the_host(run, "an int8 decision task's probabilities")


def test_an_int8_training_step_moves_the_weights_by_the_hosts_gradient():
    """One SGD step of the trainer over a decoder quantized to int8 moves its weights by the gradient
    of its loss that the host computes, from its forward's int8 products. With Triton's GEMMs open
    to XLA (`with_triton_gemms`)."""
    with_triton_gemms(training_flow)


def training_flow():
    assert_step_moves_by_the_hosts_gradient()


def test_an_int8_training_step_on_xla_default_options_moves_the_weights_by_the_hosts_gradient():
    """A run whose recorded plan compiles its step under XLA's default options, the planner's half
    rung, keeps the options its int8 products need to compute right: one SGD step moves the weights
    by the host's gradient. With Triton's GEMMs open to XLA (`with_triton_gemms`)."""
    with_triton_gemms(training_on_xla_default_options_flow)


def training_on_xla_default_options_flow():
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as records:
        os.environ["DEW_RUNG_RECORDS"] = records
        assert_step_moves_by_the_hosts_gradient()
        [path] = Path(records).glob("*.json")
        stored = json.loads(path.read_text())
        assert not stored["rung"]["xla_defaults"]
        path.write_text(json.dumps({**stored, "rung": {**stored["rung"], "xla_defaults": True}}))
        assert_step_moves_by_the_hosts_gradient()
        assert json.loads(path.read_text())["rung"]["xla_defaults"], "the run planned its own rung"


def assert_step_moves_by_the_hosts_gradient():
    """One SGD step of the trainer over a decoder quantized to int8 moves its weights by the gradient
    of its loss the host computes."""
    from dew.data import Loading, Memory
    from dew.training.trainer import Trainer

    # Eight rows, one on each of the host's eight test devices.
    spec, length = Quantization(patterns=(".*_proj",)), 4
    batch = {"text": np.random.default_rng(0).integers(0, VOCAB, (8, length + 1)).astype(np.int32)}
    held = jax.device_get(tiny().init(jax.random.key(0), jnp.ones((1, length), jnp.int32)))
    objective = LMObjective(spec.apply(tiny()), seq_len=length, ema_decay=None, variables=held)
    rows = Memory([{"text": row} for row in batch["text"]]).read("memory")
    data = rows.shuffle().repeat().batch(8).loading(Loading(workers=0, threads=1, read_buffer=1))
    state = Trainer(objective, optax.sgd(1.0), key=0).fit(data, steps=1, log_every=100)
    moved = jax.tree.map(lambda before, after: np.asarray(before, np.float64) - np.asarray(after, np.float64),
                         held["params"], jax.device_get(state.variables["params"]))

    def gradient(dtype):
        step = Step(jnp.asarray(0), jax.random.key(0), None, training=True)
        program = LMObjective(spec.apply(tiny(dtype=dtype)), seq_len=length, ema_decay=None)
        variables = placed(held, dtype)
        def loss(params):
            stats, _ = program.loss({**variables, "params": params}, {"text": jnp.asarray(batch["text"])},
                                    step)
            return program.reduce_loss(stats)[0]

        return jax.grad(loss)(variables["params"])

    def flat(tree):
        return np.concatenate([np.ravel(leaf) for leaf in jax.tree.leaves(tree)])

    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        reference = flat(jax.tree.map(lambda leaf: np.asarray(leaf, np.float64), gradient(jnp.float32)))
        with jax.enable_x64(new_val=True):
            truth = flat(jax.tree.map(lambda leaf: np.asarray(leaf, np.float64), gradient(jnp.float64)))
    assert_as_exact_as_the_reference(flat(moved), reference, truth, "an int8 training step's gradient")


@pytest.mark.parametrize("tile", [None, 16], ids=["per channel", "tiled"])
def test_a_quantized_packed_kernel_splits_into_its_members_bit_for_bit(tile):
    """A served task's packed query, key and value kernel, Qwix's quantized array in its serving wrapper,
    split by its members' widths (`split_columns`): each member dequantizes to exactly the packed
    kernel's dequantized columns, with the wrapper's recipe kept, with the input axis's scales per
    channel or tiled; Qwix's other layouts split as exactly (scales tiled along the split, one shared
    scale, a zero point, float8, the other axis), as plain host and device arrays do; a split that would
    cut a tile, or whose widths do not partition the axis, is refused. A leaf this process cannot
    materialize whole is `projections.unpacked`'s sharded tests' to cover."""
    import json
    from pathlib import Path

    import dew
    from dew.quantization import split_columns

    qwix = pytest.importorskip("qwix")
    directory = Path(__file__).parent / "fixtures/hf/llama-tiny"
    served = dew.pipeline(str(directory), dtype="float32").quantized(
        Quantization(weight_only=True, patterns=(".*_proj",), tile_size=tile))
    packed = served.variables["params"]["layers_0"]["self_attn"]["qkv_proj"]["kernel"]
    config = json.loads((directory / "config.json").read_text())
    width = packed.array.qvalue.shape[-1]
    heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
    head = width // (heads + 2 * kv_heads)
    widths = (heads * head, kv_heads * head, kv_heads * head)
    members = split_columns(packed, widths)
    whole = np.asarray(qwix.dequantize(packed.array))
    edges = np.cumsum((0, *widths))
    for member, start, stop in zip(members, edges[:-1], edges[1:], strict=True):
        assert member.how == packed.how
        np.testing.assert_array_equal(np.asarray(qwix.dequantize(member.array)), whole[..., start:stop])
    layouts = {"tiled along the split": {"qtype": jnp.int8, "tiled_axes": {1: 16}},
               "one scale shared": {"qtype": jnp.int8},
               "a zero point": {"qtype": jnp.int8, "channelwise_axes": (1,), "calibration_method": "minmax"},
               "float8": {"qtype": jnp.float8_e4m3fn, "channelwise_axes": (1,)}}
    for name, how in layouts.items():
        columns = qwix.quantize(jnp.asarray(whole), **how)
        halves = split_columns(columns, (16, width - 16))
        joined = np.concatenate([np.asarray(qwix.dequantize(half)) for half in halves], axis=1)
        np.testing.assert_array_equal(joined, np.asarray(qwix.dequantize(columns)), err_msg=name)
    rows = split_columns(qwix.quantize(jnp.asarray(whole), jnp.int8, channelwise_axes=(1,)), (16, 48), axis=0)
    assert [part.qvalue.shape[0] for part in rows] == [16, 48]
    for plain in (whole, jnp.asarray(whole)):
        np.testing.assert_array_equal(np.concatenate(split_columns(plain, widths), axis=-1), whole)
    with pytest.raises(ValueError):
        split_columns(qwix.quantize(jnp.asarray(whole), jnp.int8, tiled_axes={1: 16}), (8, width - 8))
    for wrong in ((), (16,), (16, width), (-16, width + 16)):
        with pytest.raises(ValueError):
            split_columns(packed, wrong)
