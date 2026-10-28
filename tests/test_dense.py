"""Dew's dense and convolution layers: flax's, with the bias's gradient summed at fp32.

Every configuration a layer takes initializes and computes what flax's own
layer does, bit for bit, where its fp32 sums are XLA's (`FP32_SUMS`): a CPU's own
are Dew's product policy's (`dew.nn.precision.accumulated`, tests/test_cpu_accumulation.py).
The gradient of a bias, summed over the positions it
is broadcast over, is the fp32 sum torch's addmm and convolution take, and a
biased bf16 projection's output is as close to float64 as torch's.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from reference_error import as_given, assert_as_exact, with_gradients

from dew.nn.conv import Conv
from dew.nn.dense import Dense, DenseGeneral
from dew.nn.precision import FP32_SUMS, fp32_result_dot_general
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
    """Flax's parameters bit for bit, but for a batched kernel, which draws each batch member alone
    (`test_a_batched_kernel_draws_each_member_at_its_own_fan_in`), and with flax's parameters, its
    biases zero, flax's output where both sum their fp32 products as XLA does (`FP32_SUMS`)."""
    x = jnp.asarray(np.random.default_rng(0).standard_normal(shape), dtype)
    mine, theirs = build(ours), build(flaxs)
    variables = theirs.init(jax.random.key(0), x)
    drawn = mine.init(jax.random.key(0), x)
    if getattr(theirs, "batch_dims", ()):
        # The kernel draws a member at a time; its structure, shape, dtype and box are flax's, and
        # every other leaf flax's bit for bit.
        assert jax.tree.structure(drawn) == jax.tree.structure(variables)
        for path, leaf in jax.tree_util.tree_leaves_with_path(drawn):
            other = dict(jax.tree_util.tree_leaves_with_path(variables))[path]
            if jax.tree_util.keystr(path).endswith("['kernel']"):
                assert (leaf.shape, leaf.dtype) == (other.shape, other.dtype)
            else:
                np.testing.assert_array_equal(leaf, other)
    else:
        jax.tree.map(np.testing.assert_array_equal, drawn, variables)
    if dtype == jnp.bfloat16:  # variables restored at a dtype other than the one declared
        variables = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), nn.unbox(variables))
    with jax.default_matmul_precision(FP32_SUMS):
        expected = theirs.apply(variables, x)
        got = mine.apply(variables, x)
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(expected, np.float32))


def test_a_batched_kernel_draws_each_member_at_its_own_fan_in():
    """A DenseGeneral batched over 8 groups of 512 inputs draws each group's kernel as a layer of 512
    inputs draws its own, at lecun's variance 1 / 512, where flax counts the groups as inputs too and
    draws at 1 / 4096; DeepSeek V4's grouped output projection is such a layer. Each group's 32,768
    entries pass the chi-square test of variance 1 / 512 at one in a billion each side (a normal's
    bounds; lecun's truncated normal varies its sample variance less), which flax's draw fails by
    far, the groups apart, and a partitioned initializer keeps its box."""
    from scipy import stats

    x = jnp.zeros((8, 2, 512))
    layer = DenseGeneral(64, batch_dims=(0,), use_bias=False)
    kernel = layer.init(jax.random.key(0), x)["params"]["kernel"]
    entries = kernel[0].size
    statistics = (entries - 1) * 512 * np.var(np.asarray(kernel, np.float64), axis=(1, 2), ddof=1)
    low, high = stats.chi2.ppf([1e-9, 1 - 1e-9], entries - 1)
    assert np.all((low < statistics) & (statistics < high)), (statistics, low, high)
    assert len({np.asarray(member).tobytes() for member in kernel}) == 8
    np.testing.assert_array_equal(layer.init(jax.random.key(0), x)["params"]["kernel"], kernel)
    boxed = DenseGeneral(64, batch_dims=(0,), use_bias=False, kernel_init=PARTITIONED).init(
        jax.random.key(0), x)["params"]["kernel"]
    assert isinstance(boxed, nn.Partitioned) and boxed.value.shape == (8, 512, 64)
    np.testing.assert_array_equal(boxed.value, kernel)


@pytest.mark.parametrize("kernel_init", [nn.initializers.lecun_normal(), PARTITIONED], ids=["plain", "boxed"])
def test_a_batched_kernel_of_no_members_is_empty(kernel_init):
    """A batch axis of size zero draws an empty kernel, as flax's does, and computes an empty output."""
    layer = DenseGeneral(64, batch_dims=(0,), use_bias=False, kernel_init=kernel_init)
    x = jnp.zeros((0, 2, 512))
    variables = layer.init(jax.random.key(0), x)
    assert nn.unbox(variables)["params"]["kernel"].shape == (0, 512, 64)
    assert layer.apply(variables, x).shape == (0, 2, 64)


@pytest.mark.mesh(devices=2)
def test_a_batched_kernel_draws_its_members_where_its_names_place_them():
    """A partitioned batched kernel drawn under a mesh by `jax.jit(init, out_shardings=...)` lands
    split as its names say and holds the members a single device draws."""
    from jax.sharding import Mesh, PartitionSpec

    mesh = Mesh(np.asarray(jax.devices()[:2]), ("model",))
    init = nn.with_partitioning(nn.initializers.lecun_normal(), (None, None, "model"))
    layer = DenseGeneral(64, batch_dims=(0,), use_bias=False, kernel_init=init)
    x = jnp.zeros((8, 2, 512))
    shardings = nn.get_sharding(jax.eval_shape(layer.init, jax.random.key(0), x), mesh)
    placed = jax.jit(layer.init, out_shardings=shardings)(jax.random.key(0), x)["params"]["kernel"]
    assert placed.value.sharding.spec == PartitionSpec(None, None, "model")
    np.testing.assert_array_equal(np.asarray(placed.value),
                                  np.asarray(layer.init(jax.random.key(0), x)["params"]["kernel"].value))


BIAS_SEEDS = 64
"""Check 1's declared seeds for a bias summed over every position, sized by `seeds_for` on armada: the
most any layer took."""


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

    @as_given
    def reference(x, cotangent):
        # torch sums at fp32 and rounds once into the bfloat16 bias.
        return {"bias": np.asarray(jnp.asarray(summed(cotangent, np.float32), jnp.bfloat16), np.float64)}

    assert_as_exact(dew, reference, draw, f"{layer.__class__.__name__} bias gradient", seeds=BIAS_SEEDS,
                    truth_fn=lambda x, cotangent: {"bias": summed(cotangent, np.float64)})


SPLIT = {"conv": (Conv(3, (3, 3)), (8, 32, 32, 4), 256), "dense": (Dense(3), (8, 32, 32, 4), 256)}
"""Each split bias gradient's layer, input shape and declared seeds for check 1 (`assert_as_exact`),
sized by `seeds_for` on armada: a U-ViT's output convolution, and a projection over the same positions.
Three elements a draw take 256 draws to resolve the factor."""


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("site", sorted(SPLIT))
def test_a_bias_gradient_is_as_exact_however_its_positions_are_split(site):
    """A bias's gradient sums its cotangents over every position, and is as exact with the batch
    split over eight devices as on one: check 1 over the site's seeds, truth the cotangents' float64
    sum. A CPU sums each shard's in float64 and their total before it rounds (`precision.spread`);
    a plain sum of each device's shard was 5.7 times as far from float64 (ledger 833)."""
    layer, shape, seeds = SPLIT[site]
    variables = layer.init(jax.random.key(0), jnp.zeros(shape, jnp.float32))
    out = jax.eval_shape(layer.apply, variables, jnp.zeros(shape, jnp.float32)).shape
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:8]), ("batch",))
    rows = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))
    whole = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def gradient(x, cotangent):
        forward = with_gradients(lambda params: layer.apply({"params": params}, x))
        _, grads = forward(variables["params"], cotangent)
        return {"bias": grads["bias"]}

    def draw(seed):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(shape).astype(np.float32), rng.standard_normal(out).astype(np.float32)

    split = jax.jit(gradient, in_shardings=(rows, rows), out_shardings=whole)
    alone = jax.jit(gradient)

    def truth(x, cotangent):
        return {"bias": np.asarray(cotangent, np.float64).reshape(-1, out[-1]).sum(0)}

    assert_as_exact(split, alone, draw, f"{site} bias gradient", seeds=seeds, truth_fn=truth)


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


BIASED_SEEDS = {False: 8, True: 8}
"""The biased bf16 projection's declared draws against torch's nn.Linear (check 1), Dense and
DenseGeneral."""


@pytest.mark.parametrize("general", [False, True], ids=["dense", "general"])
def test_a_biased_bf16_projection_is_as_exact_as_torchs(general):
    """A bfloat16 projection with a bias, Llama's and Qwen's query, key and value, is on par with
    torch's bfloat16 nn.Linear over the same bfloat16 weights and inputs from its float64 run over drawn
    ones (check 1): torch's addmm rounds product and bias once, Dew as flax does, the product first."""
    import torch

    layer = (DenseGeneral(32, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16) if general
             else Dense(32, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16))

    def draw(seed):
        """The inputs, the kernel and the bias, each a bfloat16 value."""
        rng = np.random.default_rng(seed)
        drawn = (rng.standard_normal((512, 64)), rng.standard_normal((64, 32)) * 0.1, rng.standard_normal(32))
        return tuple(np.asarray(jnp.asarray(value, jnp.bfloat16), np.float32) for value in drawn)

    def projected(x, kernel, bias):
        params = {"params": {"kernel": kernel.astype(jnp.bfloat16), "bias": bias.astype(jnp.bfloat16)}}
        return layer.apply(params, x.astype(jnp.bfloat16))

    def linear(dtype):
        def run(x, kernel, bias):
            weight, shift, inputs = (torch.tensor(np.asarray(a, np.float64)).to(dtype)
                                     for a in (kernel.T, bias, x))
            return torch.nn.functional.linear(inputs, weight, shift).double().numpy()
        return as_given(run)

    assert_as_exact(projected, linear(torch.bfloat16), draw, "projection", seeds=BIASED_SEEDS[general],
                    truth_fn=linear(torch.float64))


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
