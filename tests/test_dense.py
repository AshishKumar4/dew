"""Dew's dense and convolution layers: flax's, with the bias's gradient summed at fp32.

Every configuration a layer takes initializes and computes what flax's own
layer does, bit for bit. The gradient of a bias, summed over the positions it
is broadcast over, is the fp32 sum torch's addmm and convolution take, and a
biased bf16 projection's output is as close to float64 as torch's.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from reference_error import assert_as_exact, with_gradients

from dew.nn.conv import Conv
from dew.nn.dense import Dense, DenseGeneral
from dew.nn.precision import fp32_result_dot_general
from dew.quantization import Quantization, quantize_for_serving

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
    """Flax's parameters bit for bit, and with them, its biases zero, flax's output."""
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
    variables = layer.init(jax.random.key(0), jnp.zeros(shape, jnp.bfloat16))
    out = jax.eval_shape(layer.apply, variables, jnp.zeros(shape, jnp.bfloat16)).shape

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.standard_normal(shape), jnp.bfloat16),
                jnp.asarray(rng.standard_normal(out), jnp.float32).astype(jnp.bfloat16).astype(jnp.float32))

    @jax.jit
    def dew(x, cotangent):
        forward = with_gradients(lambda params: layer.apply({"params": params}, x).astype(jnp.float32))
        return {"bias": forward(variables["params"], cotangent)[1]["bias"]}

    def summed(cotangent, dtype):
        """The cotangents' sum over every position in `dtype`, as torch's addmm sums a bias's."""
        terms = np.asarray(cotangent, dtype).reshape(8192, -1)
        return terms.sum(0).reshape(variables["params"]["bias"].shape)

    def reference(x, cotangent):
        # torch sums at fp32 and rounds once into the bfloat16 bias.
        return {"bias": np.asarray(jnp.asarray(summed(cotangent, np.float32), jnp.bfloat16), np.float64)}

    assert_as_exact(dew, reference, draw, f"{layer.__class__.__name__} bias gradient", seeds=BIAS_SEEDS,
                    truth_fn=lambda x, cotangent: {"bias": summed(cotangent, np.float64)})


BIAS_SEEDS = 64
"""Check 1's declared seeds for a bias summed over every position, sized by `seeds_for` on armada: the
most any layer took."""
PROJECTION_SEEDS = 8
"""Check 1's declared seeds for a biased bfloat16 projection against torch's, sized by `seeds_for` on
armada."""
@pytest.mark.parametrize("layer", [
    Dense(8, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16),
    DenseGeneral((2, 4), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16),
    Conv(8, (3, 3), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16),
], ids=["dense", "general", "conv"])
def test_a_layer_adds_its_bias_at_its_outputs_rank(layer):
    """Under `jax_numpy_rank_promotion='raise'`, which flax's layers run under, a layer adds its
    bias reshaped to its output's rank, forward and backward, as flax's own does."""
    x = jnp.ones((2, 7, 7, 6) if isinstance(layer, Conv) else (4, 6), jnp.bfloat16)
    with jax.numpy_rank_promotion("raise"):
        variables = layer.init(jax.random.key(0), x)
        jax.grad(lambda params: jnp.sum(layer.apply({"params": params}, x).astype(jnp.float32)))(
            variables["params"])


class _Doubled:
    """A convolution hook that doubles lax's, and is false as a truth value, as a provider whose
    `__len__` is zero is."""

    def __bool__(self) -> bool:
        return False

    def __call__(self, *args, **kwargs):
        return 2 * jax.lax.conv_general_dilated(*args, **kwargs)


def test_a_convolution_runs_the_hook_it_is_given_whatever_its_truth():
    """A given `conv_general_dilated` runs, as flax runs it, even one that is false as a truth
    value: with flax's zero bias the layer's output is twice lax's."""
    x = jnp.asarray(np.random.default_rng(3).standard_normal((2, 7, 7, 6)), jnp.float32)
    plain = Conv(8, (3, 3), conv_general_dilated=None)
    variables = plain.init(jax.random.key(0), x)
    doubled = Conv(8, (3, 3), conv_general_dilated=_Doubled()).apply(variables, x)
    np.testing.assert_array_equal(np.asarray(doubled), 2 * np.asarray(plain.apply(variables, x)))


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


@pytest.mark.parametrize("general", [False, True], ids=["dense", "general"])
def test_a_biased_bf16_projection_is_as_exact_as_torchs(general):
    """A bfloat16 projection with a bias, Llama's and Qwen's query, key and value, is within
    tests/reference_error.py's rule of torch's bfloat16 nn.Linear over the same bfloat16 weights and
    inputs, both against float64: torch's addmm rounds product and bias once, Dew as flax does,
    the product first, 1.37 times torch's rounding here."""
    import torch

    layer = (DenseGeneral(32, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16) if general
             else Dense(32, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16))

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (rng.standard_normal((512, 64)).astype(np.float32),
                (rng.standard_normal((64, 32)) * 0.1).astype(np.float32),
                rng.standard_normal(32).astype(np.float32))

    @jax.jit
    def dew(x, kernel, bias):
        params = {"params": {"kernel": kernel.astype(jnp.bfloat16), "bias": bias.astype(jnp.bfloat16)}}
        return layer.apply(params, x.astype(jnp.bfloat16))

    def linear(dtype):
        def run(x, kernel, bias):
            weight, shift, inputs = (torch.tensor(np.asarray(a, np.float32)).to(torch.bfloat16).to(dtype)
                                     for a in (kernel.T, bias, x))
            return torch.nn.functional.linear(inputs, weight, shift).double().numpy()
        return run

    assert_as_exact(dew, linear(torch.bfloat16), draw, f"{'general' if general else 'dense'} projection",
                    seeds=PROJECTION_SEEDS, truth_fn=linear(torch.float64))


def test_an_adapter_adds_its_branch_to_the_layers_rounded_output():
    """A LoRA branch on a biased bfloat16 projection adds its delta to the layer's rounded output,
    as PEFT adds it to torch's: the adapted layer is the plain layer's output plus the branch alone
    (the same adapter over a zero kernel and bias), rounded once."""
    from dew.lora import Adapter, Target

    x = jnp.asarray(np.random.default_rng(2).standard_normal((4, 6)), jnp.bfloat16)
    model = _Host(Dense(8, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16,
                        bias_init=nn.initializers.normal(1.0)))
    variables = model.init(jax.random.key(0), x)
    factors = {"lora_A": jnp.asarray(np.random.default_rng(3).standard_normal((6, 2)), jnp.bfloat16),
               "lora_B": jnp.asarray(np.random.default_rng(4).standard_normal((2, 8)), jnp.bfloat16)}

    def adapted(params):
        tree = {"params": {"proj": {**params, **factors}}}
        adapter = Adapter.bound(model, tree, {("params", "proj"): Target(2, 4.0)}, rslora=False, dropout=0.0,
                                layouts={})
        return adapter.model.apply(tree, x)

    plain = model.apply(variables, x)
    branch = adapted(jax.tree.map(jnp.zeros_like, variables["params"]["proj"]))
    np.testing.assert_array_equal(np.asarray(adapted(variables["params"]["proj"]), np.float32),
                                  np.asarray(plain + branch, np.float32))


@pytest.mark.parametrize("general", [False, True], ids=["dense", "general"])
def test_a_biased_bf16_layer_quantizes_its_product(general):
    """A bfloat16 layer whose bias follows its product takes the product as flax does: Qwix
    quantizes it in training and serving, so both move off the unquantized output."""
    x = jnp.asarray(np.random.default_rng(6).standard_normal((4, 32)), jnp.bfloat16)
    biases = nn.initializers.normal(1.0)
    layer = (DenseGeneral((2, 16), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, bias_init=biases) if general
             else Dense(32, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, bias_init=biases))
    variables = layer.init(jax.random.key(0), x)
    plain = np.asarray(layer.apply(variables, x), np.float32)
    trained = np.asarray(Quantization().apply(layer).apply(variables, x), np.float32)
    model, stored = quantize_for_serving(layer, variables, Quantization(), x)
    served = np.asarray(model.apply(stored, x), np.float32)
    assert np.abs(trained - plain).max() > 0
    assert np.abs(served - plain).max() > 0


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
