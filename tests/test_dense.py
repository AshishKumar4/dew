"""Dew's dense and convolution layers: flax's, with the bias's gradient summed at fp32.

Every configuration a layer takes initializes and computes what flax's own
layer does, bit for bit; only the gradient of a bias, summed over the
positions it is broadcast over, is the fp32 sum torch's addmm and
convolution take.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from reference_error import assert_as_exact_as_the_reference

from dew.nn.conv import Conv
from dew.nn.dense import Dense, DenseGeneral
from dew.nn.precision import fp32_result_dot_general
from dew.training.quantization import Quantization, quantize_for_serving

PARTITIONED = nn.with_partitioning(nn.initializers.lecun_normal(), (None, None))
LAYERS = {
    "fp32": (lambda dense: dense(8), jnp.float32, (4, 6)),
    "bf16": (lambda dense: dense(8, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16), jnp.bfloat16, (4, 6)),
    "bf16-stored-under-fp32-declared": (lambda dense: dense(8), jnp.bfloat16, (4, 6)),
    "bf16-product-of-its-own": (
        lambda dense: dense(8, dtype=jnp.bfloat16, dot_general=jax.lax.dot_general), jnp.float32, (4, 6)),
    "fp32-product-of-its-own": (
        lambda dense: dense(8, dtype=jnp.bfloat16, dot_general=fp32_result_dot_general()),
        jnp.float32, (4, 6)),
    "no-bias": (lambda dense: dense(8, use_bias=False, dtype=jnp.bfloat16), jnp.float32, (4, 6)),
    "partitioned": (lambda dense: dense(8, kernel_init=PARTITIONED), jnp.float32, (4, 6)),
}
GENERAL = {
    "heads": (lambda general: general((2, 4), dtype=jnp.bfloat16), jnp.float32, (3, 5, 6)),
    "two-axes": (lambda general: general(4, axis=(-1, -2)), jnp.float32, (3, 5, 6)),
    "batch-dims": (lambda general: general(4, batch_dims=(0,), axis=-1), jnp.float32, (3, 5, 6)),
    "partitioned": (lambda general: general((2, 4), kernel_init=PARTITIONED), jnp.float32, (3, 5, 6)),
    "array-axes": (lambda general: general(np.array([2, 4]), axis=np.array([-1, -2])),
                   jnp.float32, (3, 5, 6)),
}
MASK = np.ones((3, 3, 6, 8), np.float32)
MASK[1, 1] = 0
CONVOLUTIONS = {
    "fp32": (lambda conv: conv(8, (3, 3)), jnp.float32, (2, 7, 7, 6)),
    "bf16": (lambda conv: conv(8, (3, 3), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16), jnp.bfloat16,
             (2, 7, 7, 6)),
    "bf16-stored-under-fp32-declared": (lambda conv: conv(8, (3, 3)), jnp.bfloat16, (2, 7, 7, 6)),
    "no-bias": (lambda conv: conv(8, (3, 3), use_bias=False, dtype=jnp.bfloat16), jnp.float32, (2, 7, 7, 6)),
    "circular": (lambda conv: conv(8, (3, 3), padding="CIRCULAR"), jnp.float32, (2, 7, 7, 6)),
    "reflect-dilated": (lambda conv: conv(8, (3, 3), padding="REFLECT", kernel_dilation=2),
                        jnp.float32, (2, 7, 7, 6)),
    "causal-depthwise": (lambda conv: conv(6, (4,), padding="CAUSAL", feature_group_count=6),
                         jnp.float32, (2, 9, 6)),
    "strided-input-dilated": (
        lambda conv: conv(8, (3, 3), strides=2, input_dilation=2, padding=((1, 1), (1, 1))),
        jnp.float32, (2, 7, 7, 6)),
    "two-batch-dims": (lambda conv: conv(8, (3, 3)), jnp.float32, (2, 3, 7, 7, 6)),
    "unbatched": (lambda conv: conv(8, (3, 3)), jnp.float32, (7, 7, 6)),
    "masked": (lambda conv: conv(8, (3, 3), mask=MASK), jnp.float32, (2, 7, 7, 6)),
    "partitioned": (lambda conv: conv(8, (3, 3), kernel_init=nn.with_partitioning(
        nn.initializers.lecun_normal(), (None, None, None, None))), jnp.float32, (2, 7, 7, 6)),
}
CASES = ([(name, Dense, nn.Dense, *case) for name, case in LAYERS.items()]
         + [(f"general-{name}", DenseGeneral, nn.DenseGeneral, *case) for name, case in GENERAL.items()]
         + [(f"conv-{name}", Conv, nn.Conv, *case) for name, case in CONVOLUTIONS.items()])


@pytest.mark.parametrize(("ours", "flaxs", "build", "dtype", "shape"), [case[1:] for case in CASES],
                         ids=[case[0] for case in CASES])
def test_a_layer_initializes_and_computes_what_flaxs_does(ours, flaxs, build, dtype, shape):
    x = jnp.asarray(np.random.default_rng(0).standard_normal(shape), dtype)
    mine, theirs = build(ours), build(flaxs)
    variables = theirs.init(jax.random.key(0), x)
    jax.tree.map(np.testing.assert_array_equal, mine.init(jax.random.key(0), x), variables)
    if dtype == jnp.bfloat16:  # variables restored at a dtype other than the one declared
        variables = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), nn.unbox(variables))
    expected = theirs.apply(variables, x)
    got = mine.apply(variables, x)
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(expected, np.float32))


@pytest.mark.parametrize(("layer", "shape"), [
    (Dense(16, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16), (8192, 12)),
    (DenseGeneral((2, 8), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16), (8192, 12)),
    (Conv(16, (3, 3), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16), (8, 32, 32, 8)),
    (Conv(16, (4,), padding="CAUSAL", feature_group_count=16, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16),
     (8, 1024, 16)),
], ids=["dense", "general", "conv", "causal-depthwise-conv"])
def test_a_bias_sums_its_gradient_over_every_position_at_fp32(layer, shape):
    """A bias broadcast over 8192 positions receives the sum of their cotangents; the sum is
    taken at fp32 and rounded once into the bfloat16 bias, as torch's addmm and convolution sum
    it, where a bfloat16 sum, which JAX gives a broadcast's transpose (jax-ml/jax#30310), lands
    several times further from float64."""
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)
    variables = layer.init(jax.random.key(0), x)
    shape = jax.eval_shape(layer.apply, variables, x).shape
    cotangent = jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)

    def loss(params):
        return jnp.sum(layer.apply({"params": params}, x).astype(jnp.float32) * cotangent.astype(jnp.float32))

    got = np.asarray(jax.grad(loss)(variables["params"])["bias"], np.float64)
    terms = np.asarray(cotangent, np.float64).reshape(8192, -1)
    assert terms.shape[1] == got.size
    truth = terms.sum(0).reshape(got.shape)
    reference = np.asarray(jnp.asarray(terms.astype(np.float32).sum(0), jnp.bfloat16), np.float64)
    assert_as_exact_as_the_reference(got, reference.reshape(got.shape), truth, "bias gradient")


def test_dense_general_refuses_an_axis_beyond_its_input_as_flaxs_does():
    x = jnp.ones((3, 6))
    for general in (DenseGeneral, nn.DenseGeneral):
        with pytest.raises((ValueError, IndexError, TypeError)):
            general(4, axis=2).init(jax.random.key(0), x)


class _Host(nn.Module):
    """A target layer carried as a submodule, as models carry one."""

    proj: nn.Module

    def __call__(self, x):
        return self.proj(x)


def test_an_adapter_adds_its_branch_to_the_output_flaxs_layer_gives():
    """A LoRA branch on a biased bfloat16 projection adds its delta to the layer's rounded
    output, as on flax's own layer and as PEFT adds it to torch's."""
    from dew.lora import Adapter, Target

    x = jnp.asarray(np.random.default_rng(2).standard_normal((4, 6)), jnp.bfloat16)
    outputs = []
    for dense in (Dense, nn.Dense):
        model = _Host(dense(8, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16,
                            bias_init=nn.initializers.normal(1.0)))
        variables = model.init(jax.random.key(0), x)
        factors = {"lora_A": jnp.asarray(np.random.default_rng(3).standard_normal((6, 2)), jnp.bfloat16),
                   "lora_B": jnp.asarray(np.random.default_rng(4).standard_normal((2, 8)), jnp.bfloat16)}
        adapted_tree = {"params": {"proj": {**variables["params"]["proj"], **factors}}}
        adapter = Adapter.bound(model, adapted_tree, {("params", "proj"): Target(2, 4.0)}, rslora=False,
                                dropout=0.0, layouts={})
        outputs.append(np.asarray(adapter.model.apply(adapted_tree, x), np.float32))
    np.testing.assert_array_equal(*outputs)


@pytest.mark.parametrize("general", [False, True], ids=["dense", "general"])
def test_a_layer_quantized_as_the_whole_model_quantizes_as_flaxs_does(general):
    """Quantized on its own, the layer its own root, a Dew layer trains and serves through
    Qwix's quantized products exactly as flax's does, and both differ from the unquantized
    layer, so neither skipped its quantization."""
    x = jnp.asarray(np.random.default_rng(5).standard_normal((4, 32)), jnp.float32)
    plain, trained, served = [], [], []
    for dense, dense_general in ((Dense, DenseGeneral), (nn.Dense, nn.DenseGeneral)):
        layer = dense_general((2, 16)) if general else dense(32)
        variables = layer.init(jax.random.key(0), x)
        plain.append(np.asarray(layer.apply(variables, x)))
        trained.append(np.asarray(Quantization().apply(layer).apply(variables, x)))
        model, stored = quantize_for_serving(layer, variables, Quantization(), x)
        served.append(np.asarray(model.apply(stored, x)))
    np.testing.assert_array_equal(*trained)
    np.testing.assert_array_equal(*served)
    assert np.abs(trained[0] - plain[0]).max() > 0
    assert np.abs(served[0] - plain[0]).max() > 0
