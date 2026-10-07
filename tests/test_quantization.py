"""Quantized training through Qwix: the value, the wrapping, the training.

Qwix-dependent tests skip without the package; the value's refusals run
everywhere, since construction never imports it.
"""

import dataclasses
import itertools
import json
from importlib import import_module

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from reference_error import assert_fp32_reduction_bound

from dew.config import OptimConfig
from dew.nn.sharding import pipeline_microbatches
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.registry import from_record, models
from dew.training.distributed import Layout, MeshSpec, shard_batch
from dew.training.quantization import Quantization, quantize_for_serving

import_module("dew.nn.backbones")  # registers the fixture kind


VOCAB = 64
SEQ_LEN = 8
BATCH = 4
TINY = {"vocab_size": VOCAB, "emb_features": 32, "num_layers": 2, "num_heads": 4,
            "num_kv_heads": 2, "mlp_features": 64, "max_seq_len": 16}


def tiny(**overrides):
    return models.build("causal_transformer", **{**TINY, **overrides})


def token_batch():
    rng = np.random.default_rng(0)
    return {"text": rng.integers(0, VOCAB, size=(BATCH, SEQ_LEN + 1)).astype(np.int32)}


def test_a_value_that_says_nothing_is_refused():
    with pytest.raises(ValueError, match="no patterns"):
        Quantization(patterns=())
    with pytest.raises(ValueError, match="does not compile"):
        Quantization(patterns=(".*(unclosed",))
    with pytest.raises(ValueError, match="calibration"):
        Quantization(calibration="percentile,99")
    with pytest.raises(ValueError, match="tile_size"):
        Quantization(tile_size=0)
    with pytest.raises(ValueError, match="bwd_qtype"):
        Quantization(bwd_qtype="int4")
    with pytest.raises(ValueError, match="bwd_stochastic_rounding"):
        Quantization(bwd_stochastic_rounding="gaussian")
    with pytest.raises(ValueError, match="int8 or fp8"):
        Quantization(dtype="int4")


def test_the_value_round_trips_through_json():
    spec = Quantization(dtype="fp8", patterns=(".*mlp.*", ".*self_attn.*"),
                        calibration="absmax,0.8", tile_size=32,
                        bwd_qtype="int8", bwd_stochastic_rounding="uniform")
    record = json.loads(json.dumps(dataclasses.asdict(spec)))
    assert from_record(Quantization, record, dtypes=False) == spec


def test_without_qwix_quantization_names_the_extra_that_installs_it(monkeypatch):
    """Quantizing without Qwix installed raises naming `dewml[quantization]`,
    where Python's own error named only the missing module."""
    import sys

    monkeypatch.setitem(sys.modules, "qwix", None)
    with pytest.raises(ModuleNotFoundError, match=r"dewml\[quantization\]"):
        Quantization().apply(tiny())


def quantized_forward(spec, **overrides):
    model = tiny(**overrides)
    variables = model.init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    qmodel = spec.apply(model)
    ids = jnp.asarray(token_batch()["text"][:, :SEQ_LEN])
    return model, qmodel, variables, ids


@pytest.mark.parametrize("scan_layers", [False, True])
def test_the_wrapped_forward_matches_qwixs_own_call(scan_layers):
    """Dew's wrapping against the provider built by hand, over the layer
    loop and over the scanned stack: identical compiled logits, and both
    away from the fp32 forward. The gap is the assertion: rules that never
    reached a matmul would leave the fp32 numerics bitwise. Observed with
    both layouts: wrapped and hand-built bitwise equal on CPU and on the RTX
    4080, 1.2e-01 (CPU) and 1.3e-01 (RTX 4080) from fp32 on logits up to 3.9."""
    qwix = pytest.importorskip("qwix")
    model, qmodel, variables, ids = quantized_forward(Quantization(), scan_layers=scan_layers)
    rules = [qwix.QtRule(module_path=".*", weight_qtype=jnp.int8,
                         act_qtype=jnp.int8)]
    reference = qwix.quantize_model(model, qwix.QtProvider(rules))
    plain = jax.jit(model.apply)(variables, ids)
    wrapped = jax.jit(qmodel.apply)(variables, ids)
    manual = jax.jit(reference.apply)(variables, ids)
    assert float(jnp.max(jnp.abs(wrapped - manual))) == 0.0
    assert float(jnp.max(jnp.abs(wrapped - plain))) > 1e-2


def test_a_pattern_that_matches_nothing_quantizes_nothing():
    """A narrowed pattern is selective: a regex no module path matches runs
    the fp32 numerics bitwise. A pattern that matched everything would move
    the logits here, so the equality fails it. Observed on CPU: bitwise
    equal."""
    pytest.importorskip("qwix")
    model, qmodel, variables, ids = quantized_forward(
        Quantization(patterns=(".*does_not_exist.*",)))
    assert float(jnp.max(jnp.abs(
        qmodel.apply(variables, ids) - model.apply(variables, ids)))) == 0.0


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


def int8_products(jaxpr) -> int:
    """How many convolutions and matmuls of two int8 operands `jaxpr` runs,
    a scan's body counted once per iteration."""
    count = 0
    for equation in jaxpr.eqns:
        if (equation.primitive.name in ("conv_general_dilated", "dot_general")
                and all(operand.aval.dtype == jnp.int8 for operand in equation.invars)):
            count += 1
        repeats = equation.params["length"] if equation.primitive.name == "scan" else 1
        for value in equation.params.values():
            inner = getattr(value, "jaxpr", value)
            if hasattr(inner, "eqns"):
                count += repeats * int8_products(inner)
    return count


def test_a_scanned_quantized_stack_quantizes_what_the_plain_one_does():
    """Quantization composes with the scan: the wrapped scan runs as many
    int8 products as the wrapped plain loop (19 for this model), and its
    compiled logits stay away from its fp32 twin, so the rules reached the
    matmuls under the scan. Its logits are checked against Qwix's own call
    over the same scanned stack (`test_the_wrapped_forward_matches_qwixs_own_call`),
    not against the plain loop's: the scan body and the unrolled layers
    compile to different fusions, which on the RTX 4080 differ by 2.3e-04
    unquantized, and int8 rounding turns that into 7.0e-02 on logits of order
    4, at the default and the highest matmul precision alike (0.0 on CPU)."""
    pytest.importorskip("qwix")
    model, qmodel, variables, ids = quantized_forward(
        Quantization(), scan_layers=True)
    plain_wrapped = Quantization().apply(tiny())
    scanned = int8_products(jax.make_jaxpr(qmodel.apply)(variables, ids).jaxpr)
    assert scanned == int8_products(jax.make_jaxpr(plain_wrapped.apply)(variables, ids).jaxpr) > 0
    reference = jax.jit(model.apply)(variables, ids)
    assert float(jnp.max(jnp.abs(jax.jit(qmodel.apply)(variables, ids) - reference))) > 1e-2


@pytest.mark.skipif(jax.default_backend() == "gpu",
                    reason="Dew refuses a grouped quantized convolution on a GPU; "
                           "test_a_gpu_refuses_grouped_quantized_convolutions covers it")
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("group_width", [1, 4])
def test_a_grouped_convolution_quantizes_each_group_on_its_own_range(group_width, dtype):
    """A grouped convolution never adds one group's inputs into another's
    outputs, so each group's activations take their own int8 scale. Over 64
    channels whose ranges span 1e-3 to 1, every output channel lands within
    int8 rounding of float (depthwise and 4 channels per group), in float32
    and in bf16 compute. The reference is the float32 convolution, at the
    highest precision, of the inputs and kernel the module computes with.
    One scale per example for all channels rounds the small groups to zero
    instead: observed on CPU before the fix, the worst channel 100%
    (depthwise) and 107% (4 per group) off. Observed after: 1.0% and 1.2%
    in float32, 1.1% and 1.2% in bf16 (1.1% and 1.4% while Qwix scaled the
    bf16 product in bf16)."""
    pytest.importorskip("qwix")
    from dew.nn.conv import Conv

    features = 64
    conv = Conv(features=features, kernel_size=(3, 3), padding="SAME",
                feature_group_count=features // group_width, use_bias=False, dtype=dtype)
    ranges = jnp.logspace(-3, 0, features)
    x = (jax.random.normal(jax.random.key(0), (2, 8, 8, features)) * ranges).astype(dtype)
    variables = jax.tree.map(
        lambda leaf: leaf.astype(dtype).astype(jnp.float32), conv.init(jax.random.key(1), x)
    )
    with jax.default_matmul_precision("highest"):
        plain = conv.clone(dtype=jnp.float32).apply(variables, x.astype(jnp.float32))
    quantized = Quantization().apply(conv).apply(variables, x)
    assert quantized.dtype == dtype
    error = (jnp.sqrt(jnp.sum((quantized.astype(jnp.float32) - plain) ** 2, axis=(0, 1, 2)))
             / jnp.sqrt(jnp.sum(plain ** 2, axis=(0, 1, 2))))
    assert float(error.max()) < 0.02, np.asarray(error)
    assert float(error.min()) > 0.0


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
    ungrouped = qwix.quantize_model(Conv(**settings), qwix.QtProvider(
        [qwix.QtRule(module_path=".*", weight_qtype=jnp.int8, act_qtype=jnp.int8)]))
    x = jax.random.normal(jax.random.key(0), (2, 8, 8, features)) * jnp.logspace(-3, 0, features)
    kernel = grouped.init(jax.random.key(1), x)["params"]["kernel"]
    cotangent = jax.random.normal(jax.random.key(2), x.shape)
    # Output channel o reads input channel i as its group's j-th: block[j, o, i].
    block = (jnp.arange(features)[None, None, :]
             == (jnp.arange(features)[None, :, None] // group_width) * group_width
             + jnp.arange(group_width)[:, None, None]).astype(kernel.dtype)
    peaks = jnp.max(jnp.abs(x.reshape(2, 8, 8, groups, group_width)), axis=(1, 2, 4))
    peaks = jnp.repeat(peaks, group_width, axis=1)[:, None, None, :]

    def dew(kernel, x):
        return Quantization().apply(grouped).apply({"params": {"kernel": kernel}}, x)

    def reference(kernel, x):
        block_diagonal = jnp.einsum("hwjo,joi->hwio", kernel, block)
        return ungrouped.apply({"params": {"kernel": block_diagonal}}, x / peaks) * peaks

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
        assert 0.0 < float(jnp.linalg.norm(got_part - float_part) / jnp.linalg.norm(float_part)) < 0.02


@pytest.mark.parametrize("dtype", ["int8", "fp8"])
def test_a_bf16_quantized_convolution_trains_as_qwix_does_in_float32(dtype):
    """A bf16 convolution with quantized activations, a hybrid DiT's patch
    embedding, computes in quantized training what Qwix's own quantized
    training computes on float32 copies of its bf16 input and kernel,
    rounded to bf16: the same value and gradients, bitwise, compiled, on
    CPU and on the RTX 4080. Before, Qwix scaled the bf16 module's 8-bit
    product in bf16, and XLA:GPU failed to compile the int8 convolution
    (`UNIMPLEMENTED: Can't lower one or more integer convolutions`)."""
    qwix = pytest.importorskip("qwix")
    from dew.nn.conv import Conv

    # Without the bias, which the quantization leaves alone and a bf16
    # module adds, and differentiates, in bf16.
    settings = {
        "features": 128,
        "kernel_size": (2, 2),
        "strides": (2, 2),
        "padding": "VALID",
        "use_bias": False,
    }
    conv = Conv(**settings, dtype=jnp.bfloat16)
    x = jax.random.normal(jax.random.key(0), (8, 16, 16, 4), jnp.bfloat16)
    variables = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), conv.init(jax.random.key(1), x))
    cotangent = jax.random.normal(jax.random.key(2), (8, 8, 8, 128), jnp.bfloat16)
    qtype = jnp.int8 if dtype == "int8" else jnp.float8_e4m3fn
    reference = qwix.quantize_model(Conv(**settings), qwix.QtProvider(
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
    reference = qwix.quantize_model(conv, qwix.QtProvider([qwix.QtRule(
        module_path=".*", weight_qtype=jnp.int8, op_names=("dot_general", "einsum", "dot"))]))

    def value_and_gradients(module):
        out, transpose = jax.vjp(jax.jit(module.apply), variables, x)
        return out, *transpose(cotangent)

    got = value_and_gradients(Quantization(weight_only=True).apply(conv))
    assert got[0].dtype == jnp.bfloat16
    jax.tree.map(np.testing.assert_array_equal, got, value_and_gradients(reference))
    jax.tree.map(np.testing.assert_array_equal, got, value_and_gradients(conv))


@pytest.mark.skipif(jax.default_backend() == "gpu",
                    reason="Dew refuses a grouped quantized convolution on a GPU; "
                           "test_a_gpu_refuses_grouped_quantized_convolutions covers it")
def test_quantized_gradients_of_a_grouped_convolution_are_refused():
    """Qwix 0.1.8 cannot quantize a grouped convolution's gradients, so
    training one with `bwd_qtype` raises naming the ways around it."""
    pytest.importorskip("qwix")
    from dew.nn.conv import Conv

    conv = Conv(features=16, kernel_size=(3, 3), padding="SAME", feature_group_count=16, use_bias=False)
    x = jax.random.normal(jax.random.key(0), (2, 8, 8, 16))
    variables = conv.init(jax.random.key(1), x)
    with pytest.raises(ValueError, match=r"bwd_qtype.*spatial_fusion"):
        Quantization(bwd_qtype="int8").apply(conv).apply(variables, x)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="needs a GPU")
@pytest.mark.parametrize("dtype", ["int8", "fp8"])
@pytest.mark.parametrize("group_width", [1, 4])
@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_a_gpu_refuses_grouped_quantized_convolutions(group_width, dtype, dilation):
    """XLA:GPU computes a grouped convolution with int8 or fp8 activations
    wrongly or not at all, depending on the GPU, so quantizing one raises
    naming the ways around it, for training and for serving alike; weight-only
    quantization, one of them, leaves the convolution in float."""
    pytest.importorskip("qwix")
    from dew.nn.conv import Conv

    features = 64
    conv = Conv(features=features, kernel_size=(3, 3), padding="SAME",
                feature_group_count=features // group_width, use_bias=False,
                kernel_dilation=(dilation, dilation))
    x = jax.random.normal(jax.random.key(0), (2, 8, 8, features))
    variables = conv.init(jax.random.key(1), x)
    with pytest.raises(ValueError, match="spatial_fusion"):
        Quantization(dtype=dtype).apply(conv).apply(variables, x)
    with pytest.raises(ValueError, match="spatial_fusion"):
        quantize_for_serving(conv, variables, Quantization(dtype=dtype), x)
    served, served_variables = quantize_for_serving(
        conv, variables, Quantization(dtype=dtype, weight_only=True), x
    )
    np.testing.assert_array_equal(served.apply(served_variables, x), conv.apply(variables, x))


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
    direct = qwix.quantize_model(block, qwix.QtProvider(
        [qwix.QtRule(module_path=pattern, weight_qtype=jnp.int8, act_qtype=jnp.int8)]))

    def value_and_gradients(module, variables):
        out, transpose = jax.vjp(jax.jit(module.apply), variables, x)
        return out, *transpose(cotangent)

    plain = value_and_gradients(block, variables)
    jax.tree.map(np.testing.assert_array_equal, value_and_gradients(spec.apply(block), variables), plain)
    jax.tree.map(np.testing.assert_array_equal, value_and_gradients(direct, variables), plain)
    served, served_variables = quantize_for_serving(block, variables, spec, x)
    np.testing.assert_array_equal(jax.jit(served.apply)(served_variables, x), plain[0])


def test_serving_stores_int8_kernels_and_computes_what_training_quantized():
    """Served weights are int8 values with their scales, and the served
    projections reproduce the quantized-training forward they were trained
    under, away from fp32. Observed on CPU: served against wrapped 0.0 on
    logits of order 3, 1.2e-01 from fp32. (Qwix's serving and training
    providers quantize attention's matmuls of two activations differently:
    with those quantized too, the two forwards differ by 3e-02.)"""
    pytest.importorskip("qwix")
    spec = Quantization(patterns=(".*_proj",))
    model, qmodel, variables, ids = quantized_forward(spec)
    served, served_variables = quantize_for_serving(model, variables, spec, ids)
    dtypes = {leaf.dtype for leaf in jax.tree.leaves(served_variables["params"])}
    assert jnp.dtype(jnp.int8) in dtypes
    logits = served.apply(served_variables, ids)
    assert float(jnp.max(jnp.abs(logits - qmodel.apply(variables, ids)))) < 1e-4
    assert float(jnp.max(jnp.abs(logits - model.apply(variables, ids)))) > 1e-2


@pytest.mark.parametrize("weight_only", [False, True])
def test_a_served_language_model_generates_with_its_quantized_kernels(weight_only):
    """Generation enters a language model through methods besides
    `__call__` (`states_and_logits_at` for the prefill, `states_and_logits`
    for each step), and they compute with the stored quantized kernels as
    `__call__` does. Before, they ran outside Qwix's interception and a
    stored kernel reached a Dense raw (`TypeError: expected number, got
    WithAux`)."""
    pytest.importorskip("qwix")
    from dew.sampling import Sampling, generate

    model = tiny()
    prompt = token_batch()["text"][:, :4]
    variables = model.init(jax.random.key(0), prompt)
    served, served_variables = quantize_for_serving(
        model, variables, Quantization(weight_only=weight_only), prompt
    )
    logits = served.apply(served_variables, prompt)
    _, stepped = served.apply(served_variables, prompt, method="states_and_logits")
    np.testing.assert_array_equal(stepped, logits)
    assert float(jnp.max(jnp.abs(logits - model.apply(variables, prompt)))) > 1e-3
    tokens = (
        generate(served, served_variables, prompt, 3, key=0, sampling=Sampling(temperature=0)).host().tokens
    )
    assert tokens.shape == (BATCH, 7)


@pytest.mark.parametrize("dtype", ["int8", "fp8"])
@pytest.mark.parametrize("weight_only", [False, True])
def test_a_quantized_text_task_matches_qwix_direct_logits_and_generation(dtype, weight_only):
    """Qwix 0.1.8 PTQ with fp32 compute, compared bitwise with the public task.

    Greedy agreement with bf16 compute over the same fp32 master weights is
    reported separately; quantization changes the policy.
    """
    import functools

    from dew.inference import TextGeneration
    from dew.sampling import Sampling
    from dew.training.quantization import METHODS

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
    reference = qwix.quantize_model(model, qwix.PtqProvider(rules),
                                    methods=tuple(method for method in METHODS if hasattr(model, method)))
    abstract = jax.eval_shape(functools.partial(reference.init, jax.random.key(0),
                                               jnp.zeros((1, 1), jnp.int32)))
    expected_variables = {
        **variables,
        "params": qwix.quantize_params(variables["params"], abstract["params"]),
    }
    expected_logits = jax.jit(reference.apply)(expected_variables, prompt)
    logits = jax.jit(served.model.apply)(served.variables, prompt)
    np.testing.assert_array_equal(logits, expected_logits)
    assert float(jnp.max(jnp.abs(logits - model.apply(variables, prompt)))) > 1e-3
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


@pytest.fixture(scope="module")
def flux_task(tmp_path_factory):
    """The tiny FLUX source's text-to-image task: a matmul-only denoiser
    beside CLIP and T5 towers and a VAE."""
    import tarfile
    from pathlib import Path

    from dew.interop.pretrained import Pretrained

    root = tmp_path_factory.mktemp("flux")
    with tarfile.open(Path(__file__).parent / "fixtures/flux_source.tar.xz") as archive:
        archive.extractall(root, filter="data")
    return Pretrained.load(root / "pipeline", dtype="float32", attention_impl="xla").text_to_image()


@pytest.mark.parametrize("dtype", ["int8", "fp8"])
@pytest.mark.parametrize("weight_only", [False, True])
def test_a_quantized_image_task_matches_qwix_direct_prediction_and_samples(flux_task, dtype, weight_only):
    """Qwix 0.1.8 PTQ over the task's denoiser, built here from Qwix's own
    rules, provider and `quantize_params`, is the reference for
    `TextToImage.quantized`: the same quantized kernels, the denoiser's
    prediction bitwise, and a decoded two-step sample bitwise. The towers
    and the VAE keep their weights, leaf for leaf."""
    import functools

    from dew.training.quantization import METHODS

    qwix = pytest.importorskip("qwix")
    task = flux_task
    served = task.quantized(Quantization(dtype=dtype, weight_only=weight_only))
    qtype = jnp.int8 if dtype == "int8" else jnp.float8_e4m3fn
    fields = {"op_names": ("dot_general", "einsum", "dot")} if weight_only else {}
    rules = [qwix.QuantizationRule(module_path=".*", weight_qtype=qtype,
                                  act_qtype=None if weight_only else qtype, **fields)]
    reference = qwix.quantize_model(task.model, qwix.PtqProvider(rules),
                                    methods=tuple(name for name in METHODS if hasattr(task.model, name)))
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
    assert float(jnp.abs(prediction - unquantized).max()) > 1e-3
    direct = dataclasses.replace(task, model=reference, variables={**task.variables, **expected})
    np.testing.assert_array_equal(served(["a red bird"], steps=2, key=1).host().images,
                                  direct(["a red bird"], steps=2, key=1).host().images)


@pytest.mark.mesh
def test_quantizing_resident_weights_keeps_their_parameter_shards():
    from dew.inference import TextGeneration

    pytest.importorskip("qwix")
    model = tiny()
    prompt = jnp.asarray(token_batch()["text"][:, :4])
    variables = model.init(jax.random.key(0), prompt)
    mesh = MeshSpec(fsdp=2).build()
    variables = jax.device_put(variables, Layout(min_shard=1).shardings(mesh, variables))
    served = TextGeneration(model, variables).quantized(Quantization(weight_only=True, patterns=(".*_proj",)))
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        original = variables["params"]["layers_0"]["self_attn"][name]["kernel"]
        quantized = served.variables["params"]["layers_0"]["self_attn"][name]["kernel"].array.qvalue
        assert "fsdp" in str(original.sharding.spec)
        assert quantized.sharding == original.sharding


def test_quantizing_host_weights_keeps_numpy_storage_and_matches_qwix():
    """Host kernels use Qwix 0.1.8 on the active backend and retain host storage."""
    import functools

    from dew.inference import TextGeneration
    from dew.sampling import Sampling
    from dew.training.quantization import METHODS

    qwix = pytest.importorskip("qwix")
    model = tiny()
    prompt = jnp.asarray(token_batch()["text"][:, :4])
    resident = model.init(jax.random.key(0), prompt)
    host = jax.device_get(resident)
    spec = Quantization(weight_only=True, patterns=(".*_proj",))
    task = TextGeneration(model, host, sampling=Sampling(temperature=0))
    served = task.quantized(spec)
    assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(served.variables))
    reference = qwix.quantize_model(model, qwix.PtqProvider([
        qwix.QuantizationRule(module_path=".*_proj", weight_qtype=jnp.int8,
                              op_names=("dot_general", "einsum", "dot"))]),
        methods=tuple(method for method in METHODS if hasattr(model, method)))
    abstract = jax.eval_shape(functools.partial(reference.init, jax.random.key(0),
                                               jnp.zeros((1, 1), jnp.int32)))
    reference_variables = {**host, "params": qwix.quantize_params(resident["params"], abstract["params"])}
    expected = TextGeneration(reference, jax.device_get(reference_variables), sampling=task.sampling)
    np.testing.assert_array_equal(jax.jit(served.model.apply)(served.variables, prompt),
                                  jax.jit(expected.model.apply)(expected.variables, prompt))
    np.testing.assert_array_equal(served(prompt, 3, key=7).host().tokens,
                                  expected(prompt, 3, key=7).host().tokens)


def test_quantizing_packed_pipeline_weights_matches_the_canonical_quantized_model():
    from pathlib import Path

    import dew
    from dew.inference.serving import Server
    from dew.interop import Pretrained
    from dew.sampling import Sampling

    pytest.importorskip("qwix")
    directory = Path(__file__).parent / "fixtures/hf/llama-tiny"
    spec = Quantization(weight_only=True, patterns=(".*_proj",))
    canonical = Pretrained.load(directory, dtype="float32").text_generation(
        sampling=Sampling(temperature=0, eos_id=None))
    packed = dew.pipeline(str(directory), dtype="float32")
    reference = canonical.quantized(spec)
    served = packed.quantized(spec)
    tokens = jnp.asarray([[1, 2, 3]], jnp.int32)
    np.testing.assert_allclose(served.model.apply(served.variables, tokens),
                               reference.model.apply(reference.variables, tokens), atol=2e-6, rtol=2e-6)
    expected = reference(tokens, 4, key=5).host()
    actual = served(tokens, 4, key=5).host()
    np.testing.assert_array_equal(actual.tokens, expected.tokens)
    np.testing.assert_allclose(actual.raw_log_probs, expected.raw_log_probs, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(actual.behavior_log_probs, expected.behavior_log_probs, atol=2e-6, rtol=2e-6)
    server = Server.from_task(served, slots=max(2, jax.device_count()), capacity=64)
    drawn = server([[1, 2, 3]], 4, key=5)[0].host()
    np.testing.assert_array_equal(drawn.tokens, expected.tokens)
    np.testing.assert_allclose(drawn.raw_log_probs, expected.raw_log_probs, atol=2e-6, rtol=2e-6)


def test_a_quantized_text_task_without_qwix_names_the_install_extra(monkeypatch):
    import sys

    from dew.inference import TextGeneration

    monkeypatch.setitem(sys.modules, "qwix", None)
    with pytest.raises(ModuleNotFoundError, match=r"dewml\[quantization\]"):
        TextGeneration(tiny(), {}).quantized(Quantization())


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


@pytest.mark.parametrize("weight_only", [False, True])
def test_a_served_bf16_module_computes_in_bf16(weight_only):
    """A quantized kernel follows the module's compute dtype as its float
    kernel would: a bf16 Dense stays bf16 through its matmul. Before, the
    fp32 scales of the stored kernel promoted it to fp32."""
    pytest.importorskip("qwix")
    from flax import linen as nn

    dense = nn.Dense(16, dtype=jnp.bfloat16)
    x = jax.random.normal(jax.random.key(0), (4, 32), jnp.bfloat16)
    variables = dense.init(jax.random.key(1), x)
    served, served_variables = quantize_for_serving(
        dense, variables, Quantization(weight_only=weight_only), x
    )
    assert served.apply(served_variables, x).dtype == jnp.bfloat16


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


def converted_int32_products(jaxpr) -> set:
    """The dtypes the int32 results of the convolutions and matmuls in
    `jaxpr`, and in the jaxprs it calls, are converted to."""
    products, dtypes = set(), set()
    for equation in jaxpr.eqns:
        if (equation.primitive.name in ("conv_general_dilated", "dot_general")
                and equation.outvars[0].aval.dtype == jnp.int32):
            products.add(id(equation.outvars[0]))
        if equation.primitive.name == "convert_element_type" and id(equation.invars[0]) in products:
            dtypes.add(jnp.dtype(equation.params["new_dtype"]))
        for value in equation.params.values():
            inner = getattr(value, "jaxpr", value)
            if hasattr(inner, "eqns"):
                dtypes |= converted_int32_products(inner)
    return dtypes


@pytest.mark.skipif(jax.default_backend() == "gpu",
                    reason="Dew refuses a grouped quantized convolution on a GPU; "
                           "test_a_gpu_refuses_grouped_quantized_convolutions covers it")
@pytest.mark.parametrize("training", [True, False])
def test_a_bf16_int8_depthwise_convolution_scales_its_int32_products_in_float32(training):
    """A bf16 depthwise convolution quantized to int8 converts its int32
    product to float32 for the scales, in training as in serving, and
    returns bf16. Qwix 0.1.8 converted it to bf16 in training, and on a v6e
    XLA:TPU computed an int8 depthwise convolution scaled that way as NaN in
    all but a few outputs."""
    pytest.importorskip("qwix")
    from dew.nn.conv import Conv

    features = 16
    conv = Conv(features=features, kernel_size=(3, 3), padding="SAME", feature_group_count=features,
                use_bias=False, dtype=jnp.bfloat16)
    x = jax.random.normal(jax.random.key(0), (2, 8, 8, features), jnp.bfloat16)
    variables = conv.init(jax.random.key(1), x)
    if training:
        module, module_variables = Quantization().apply(conv), variables
    else:
        module, module_variables = quantize_for_serving(conv, variables, Quantization(), x)
    assert converted_int32_products(jax.make_jaxpr(module.apply)(module_variables, x).jaxpr) == {
        jnp.dtype(jnp.float32)}
    assert module.apply(module_variables, x).dtype == jnp.bfloat16


def test_quantized_training_differentiates_through_complex_matmuls_in_float():
    """Quantized training leaves a matmul with a complex operand, like the
    S5 scan's in the hybrid DiT, in float, and differentiates through it:
    values and gradients are those of Qwix's own quantized training applied
    to the real Dense alone. Observed on CPU: bitwise equal, 2.1e-02 from fp32
    in the kernel's gradient. Before, Qwix quantized the real operand of a
    real-by-complex matmul, its backward returned a complex gradient for
    that real operand, and `jax.grad` failed on it."""
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
    reference = qwix.quantize_model(model, qwix.QtProvider(
        [qwix.QtRule(module_path="proj", weight_qtype=jnp.int8, act_qtype=jnp.int8)]))

    def value_and_gradients(module):
        return jax.jit(jax.value_and_grad(lambda v, x: jnp.sum(module.apply(v, x) ** 2), argnums=(0, 1)))(
            variables, x)

    got = value_and_gradients(Quantization().apply(model))
    want = value_and_gradients(reference)
    jax.tree.map(np.testing.assert_array_equal, got, want)
    _, (kernel_gradient, _) = value_and_gradients(model)
    error = jnp.linalg.norm(
        got[1][0]["params"]["proj"]["kernel"] - kernel_gradient["params"]["proj"]["kernel"]
    )
    assert 0.0 < float(error / jnp.linalg.norm(kernel_gradient["params"]["proj"]["kernel"])) < 0.05


def test_serving_leaves_complex_matmuls_in_float():
    """Qwix quantizes real values only; a complex matmul, like the S5 scan's
    in the hybrid DiT, runs unquantized beside the quantized Dense, within
    int8 rounding of float. Observed on CPU: 4e-03 relative."""
    pytest.importorskip("qwix")
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
    plain = model.apply(variables, x)
    error = jnp.linalg.norm(served.apply(served_variables, x) - plain) / jnp.linalg.norm(plain)
    assert 0.0 < float(error) < 0.02


@pytest.mark.mesh
def test_a_quantized_pipeline_has_finite_loss_and_gradients():
    """The int8 trunk over two stages on the eight simulated devices: finite
    loss and gradients, and the loss within int8 rounding of the plain
    wrapped loop's, the way the bf16 pipeline test holds its policy's
    rounding. Observed on CPU: 1.6e-03 on a loss of order 5."""
    pytest.importorskip("qwix")
    model = tiny(num_layers=4)
    variables = model.init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    qmodel = Quantization().apply(model)
    rows = np.random.default_rng(0).integers(0, VOCAB, size=(8, SEQ_LEN + 1)).astype(np.int32)
    batch = {"text": rows}

    def run(spec):
        objective = LMObjective(qmodel, SEQ_LEN)
        mesh = spec.build()
        placed = jax.device_put(
            variables, Layout(min_shard=256).shardings(mesh, variables))
        placed_batch = shard_batch(mesh, batch)
        info = Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(3),
                    ema=None)

        def loss(params):
            return objective.scalar_loss({**variables, "params": params},
                                  placed_batch, info)

        with jax.set_mesh(mesh), pipeline_microbatches(spec.microbatches):
            (value, _), grads = jax.jit(
                jax.value_and_grad(loss, has_aux=True))(placed["params"])
        return float(value), jax.tree.map(np.asarray, grads)

    loss, _grads = run(MeshSpec(fsdp=8))
    piped, piped_grads = run(MeshSpec(fsdp=4, stage=2, microbatches=2))
    assert np.isfinite(loss) and np.isfinite(piped)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(piped_grads))
    assert abs(loss - piped) < 1e-2 * max(1.0, abs(loss)), (loss, piped)


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
    from dew.config import ModelConfig, TrainerConfig
    from dew.data import TFDSImages
    from dew.objectives.diffusion import DiffusionRunConfig

    return DiffusionRunConfig(
        model=ModelConfig("simple_dit", {"patch_size": 4, "emb_features": 16,
                                         "num_layers": 1, "num_heads": 2, "dtype": "float32"}),
        data=TFDSImages(image_size=RES), text=None, guidance=None,
        sampling_steps=2, val_metrics=(),
        trainer=TrainerConfig(name="quantized", checkpoint_dir=str(directory), batch_size=batch,
                              steps=1, eval_every=None, checkpoint_every=None,
                              compilation_cache_dir=None, **trainer))


def test_the_trainer_knob_quantizes_the_objective_a_run_trains(tmp_path):
    """`--trainer.quantization` is the one place a run names quantization:
    `RunConfig.train` wraps the module the objective trains before anything
    initialises it, and the run steps on the quantized forward. The distance
    from the fp32 forward on the trained weights is the assertion that the
    rules reached the matmuls. Observed on CPU: 2.7e-05 on activations of
    order 1e-2, against 0.0 for a run trained without the knob."""
    pytest.importorskip("qwix")
    from dew.data import Dataset

    batch = RUN_BATCH
    config = diffusion_run(tmp_path, batch, quantization=Quantization())
    objective = config.build()
    plain = config.build()

    state = config.train(objective, Dataset(lambda partition: image_batches(batch)(), None, None, batch),
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
    from dew.training.quantization import _quantize

    def decoder():
        return CausalTransformer(vocab_size=32, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                                 max_seq_len=8)

    student, teacher = decoder(), decoder()
    objective = DistillationObjective(LMObjective(student, seq_len=4), LMObjective(teacher, seq_len=4))
    _quantize(objective, Quantization())
    tokens = jnp.arange(8, dtype=jnp.int32)[None]
    variables = student.init(jax.random.key(0), tokens)
    trained, frozen = (entry.module for entry in objective.program_key())
    assert float(jnp.max(jnp.abs(jax.jit(trained.apply)(variables, tokens)
                                 - jax.jit(student.apply)(variables, tokens)))) > 0.0
    np.testing.assert_array_equal(jax.jit(frozen.apply)(variables, tokens),
                                  jax.jit(teacher.apply)(variables, tokens))


def test_an_objective_that_trains_no_single_model_is_refused(tmp_path):
    """The wrap needs a module to wrap; an objective that keeps none is
    refused by name, before the run writes anything."""
    from dew.data import Dataset
    from dew.objectives.base import Aux, Objective

    class Modelless(Objective):
        def init(self, key, variables=None):
            return {"params": {}}

        def loss(self, variables, batch, step):
            return jnp.zeros(()), Aux({})

    config = diffusion_run(tmp_path, quantization=Quantization())
    with pytest.raises(ValueError, match="Modelless trains none"):
        config.train(
            Modelless(),
            Dataset(lambda partition: image_batches(RUN_BATCH)(), None, None, RUN_BATCH),
            name="quantized",
        )
    assert not (tmp_path / "quantized").exists()
