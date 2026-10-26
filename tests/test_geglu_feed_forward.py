"""diffusers' GEGLU feed-forward as a packed `GatedMLP` (`geglu_feed_forward`): its one projection
holds the up half first, as the checkpoint stores it, and computes what diffusers' `FeedForward`
computes from it, on par with that computation from its float64 run (check 1)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import as_given, assert_as_exact, with_gradients

from dew.diffusion.process import DenoisingCondition
from dew.nn.activations import gated_product
from dew.nn.dense import Dense
from dew.nn.mlp import GatedMLP, geglu_feed_forward
from dew.nn.precision import float32_dot_general
from dew.nn.sharding import declarations, declared_axes, parameter_path

GEGLU_SEEDS = {"tanh float32": 128, "tanh bfloat16": 128, "erf float32": 128, "erf bfloat16": 128}
"""Check 1's declared seeds for each form, sized by `seeds_for` on armada."""


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["float32", "bfloat16"])
@pytest.mark.parametrize("approximate", [True, False], ids=["tanh", "erf"])
def test_the_packed_feed_forward_computes_diffusers_geglu(approximate, dtype):
    """diffusers' `GEGLU`: one projection whose first half is the hidden state and second the gate,
    `hidden * gelu(gate)`, then the output map. The packed GatedMLP's output and every gradient are on
    par with that computation's in the same dtype (check 1), from its float64 run, over drawn states,
    parameters and cotangents."""
    dim = 8
    shape = (4, 16, dim)
    packed = geglu_feed_forward(dim, approximate=approximate, dtype=dtype)
    shapes = jax.eval_shape(packed.init, jax.random.key(0), jnp.zeros(shape, dtype))["params"]

    def draw(seed):
        rng = np.random.default_rng(seed)
        params = jax.tree.map(lambda leaf: np.asarray(rng.standard_normal(leaf.shape) * 0.3, leaf.dtype),
                              shapes)
        x, cotangent = (np.asarray(rng.standard_normal(shape), dtype) for _ in range(2))
        return params, x, cotangent

    def diffusers(dtype):
        dense = {width: Dense(width, dtype=dtype, dot_general=float32_dot_general)
                 for width in (8 * dim, dim)}

        def forward(params, x):
            hidden, gate = jnp.split(dense[8 * dim].apply({"params": params["gate_up_proj"]}, x), 2, axis=-1)
            activated = gated_product("geglu" if approximate else "geglu_exact")(gate, hidden)
            return dense[dim].apply({"params": params["down_proj"]}, activated)
        return jax.jit(with_gradients(forward))

    label = f"{'tanh' if approximate else 'erf'} {jnp.dtype(dtype).name}"
    assert_as_exact(jax.jit(with_gradients(lambda params, x: packed.apply({"params": params}, x))),
                    diffusers(dtype), draw, label, seeds=GEGLU_SEEDS[label], truth_fn=diffusers(None))


GELU_TANH_SEEDS = 8
"""Check 1's declared seeds for the tanh GELU, sized by `seeds_for` on armada."""


def test_the_tanh_gelu_and_its_gradient_are_as_exact_as_torchs():
    """The tanh GELU (`gelu_tanh`, Hugging Face's gelu_pytorch_tanh) in fp32: its value and its input's
    gradient under a drawn cotangent are on par with torch's `gelu(approximate='tanh')` from its float64
    run (check 1), over drawn inputs at scale 4, which reach both tails: where the positive tail's
    sigmoid rounds to 1, from x near 5, the gradient keeps the term autodiff of the sigmoid drops."""
    import torch
    from reference_error import torch_with_gradients

    from dew.nn.activations import gelu_tanh

    def draw(seed):
        rng = np.random.default_rng([0x6E, seed])
        return tuple((scale * rng.standard_normal((64, 256))).astype(np.float32) for scale in (4, 1))

    def named(run):
        """`run`'s value and gradient by name."""
        return as_given(lambda *inputs: dict(zip(("value", "gradient"), run(*inputs), strict=True)))

    def theirs(dtype):
        pulled = torch_with_gradients(lambda x: torch.nn.functional.gelu(x, approximate="tanh"))
        return named(lambda x, cotangent: pulled(np.asarray(x, dtype), np.asarray(cotangent, dtype)))

    assert_as_exact(named(jax.jit(with_gradients(gelu_tanh))), theirs(np.float32), draw, "the tanh gelu",
                    seeds=GELU_TANH_SEEDS, truth_fn=theirs(np.float64))


def test_a_unets_feed_forward_declares_its_kernels_axes():
    """Every leaf of a UNet's feed-forward, the packed projection's included, carries declared axes:
    the packed projection's kernel `embed` to `mlp` and its bias `mlp`, so a tensor axis splits it as
    it splits any MLP."""
    from dew.registry import models

    model = models.build("unet_2d_condition", stages=[{"features": 32, "heads": 2}], blocks_per_level=1)
    params = jax.eval_shape(lambda: model.init(
        jax.random.key(0), jnp.zeros((1, 8, 8, 4)), jnp.ones((1,)),
        conditioning=DenoisingCondition(jnp.zeros((1, 3, 768)))))["params"]
    declared = declarations(params)
    leaves = [(path, leaf) for path, leaf in jax.tree_util.tree_leaves_with_path(params)
              if "ff" in parameter_path(path)]
    assert {parameter_path(path)[-2] for path, _ in leaves} == {"gate_up_proj", "down_proj"}
    expected = {"kernel": ("embed", "mlp"), "bias": ("mlp",)}
    for path, leaf in leaves:
        axes = declared_axes(path, len(leaf.shape), declared)
        names = parameter_path(path)
        assert axes is not None, names
        if names[-2] == "gate_up_proj":
            assert axes == expected[names[-1]], names


def test_a_packed_order_names_the_gate_and_up_projections():
    """A packed order other than the two projections' is refused."""
    mlp = GatedMLP(hidden_features=8, out_features=4, activation="geglu", packed=True,
                   packed_order=("gate_proj", "down_proj"))
    with pytest.raises(ValueError):
        mlp.init(jax.random.key(0), jnp.zeros((1, 4)))


def test_a_transformer_with_dropout_trains_and_evaluates_without_a_dropout_key():
    """A nonzero-dropout spatial transformer trains with sampled masks and evaluates with them
    disabled. Its update reaches both attentions and the feed-forward; evaluation needs no RNG."""
    from dew.nn.attention import BasicTransformerBlock

    block = BasicTransformerBlock(8, heads=2, dim_head=4, norm="layer", dropout_rate=0.1,
                                  attention_impl="reference")
    x = jax.random.normal(jax.random.key(1), (2, 4, 8))
    context = jax.random.normal(jax.random.key(2), (2, 3, 8))
    variables = block.init({"params": jax.random.key(3), "dropout": jax.random.key(4)}, x, context,
                           train=True)

    def loss(params, key):
        output = block.apply({"params": params}, x, context, train=True, rngs={"dropout": key})
        return jnp.mean(jnp.square(output))

    value, gradient = jax.value_and_grad(loss)(variables["params"], jax.random.key(5))
    assert np.isfinite(value)
    for name in ("Attention1", "Attention2", "ff"):
        leaves = jax.tree.leaves(gradient[name])
        assert all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)
        assert any(np.any(np.asarray(leaf) != 0) for leaf in leaves)
    updated = jax.tree.map(lambda weight, grad: weight - 0.01 * grad, variables["params"], gradient)
    evaluated = block.apply({"params": updated}, x, context, train=False)
    assert np.isfinite(np.asarray(evaluated)).all()
    sampled = [block.apply({"params": updated}, x, context, train=True,
                           rngs={"dropout": jax.random.key(seed)})
               for seed in (6, 7)]
    assert np.any(np.asarray(sampled[0]) != np.asarray(sampled[1]))
