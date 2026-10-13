"""LoRA's rsLoRA scaling and its branch dropout in training against PEFT
0.20.0's own forward and backward (`tools/lora_peft_reference.py`), and a
target that contracts and produces several axes against a float64 oracle
of its forward and its factors' gradient."""


import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from reference_error import assert_as_exact_as_the_reference

from dew.lora import Adapter, Target


class Host(nn.Module):
    """One DenseGeneral that contracts the input's last two axes, listed out
    of order, into a (5, 6) output."""

    @nn.compact
    def __call__(self, x):
        return nn.DenseGeneral((5, 6), axis=(-1, -2), name="proj")(x)


def test_a_multi_axis_target_steps_as_its_flattened_matrix_does():
    """The branch of a target contracting (8, 6) input axes into (5, 6)
    output ones is the matrix LoRA of the flattened kernel: with x as
    [batch, 48] in the kernel's sorted axis order, A as [48, r] and B as
    [r, 30], the output is x K + bias + s x A B, and a fixed cotangent G
    gives A's gradient s x^T G B^T and B's s (x A)^T G. The float64 oracle
    computes that flattened form; numpy in float32 is the reference whose
    rounding Dew's forward and gradients are held to by the float64 rule,
    over 480, 192 and 120 entries."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(16, 6, 8)).astype(np.float32)
    model = Host()
    params = jax.tree.map(np.asarray, model.init(jax.random.key(0), jnp.asarray(x)))["params"]
    target = Target(rank=4, alpha=3.0)
    a = rng.normal(size=(6, 8, 4)).astype(np.float32)
    b = rng.normal(size=(4, 5, 6)).astype(np.float32)
    cotangent = rng.normal(size=(16, 5, 6)).astype(np.float32)
    tree = {"params": {"proj": {**params["proj"], "lora_A": a, "lora_B": b}}}
    adapter = Adapter.bound(model, tree, {("params", "proj"): target}, rslora=False, dropout=0.0, layouts={})
    scale = adapter.scale(target)

    def output(factors):
        node = {**params["proj"], **factors}
        return adapter.model.apply({"params": {"proj": node}}, jnp.asarray(x))

    def functional(factors):
        return jnp.sum(output(factors) * cotangent)

    forward = output({"lora_A": a, "lora_B": b})
    gradient = jax.grad(functional)({"lora_A": jnp.asarray(a), "lora_B": jnp.asarray(b)})

    def oracle(dtype):
        rows = x.reshape(16, 48).astype(dtype)
        kernel = params["proj"]["kernel"].reshape(48, 30).astype(dtype)
        down, up = a.reshape(48, 4).astype(dtype), b.reshape(4, 30).astype(dtype)
        seed = cotangent.reshape(16, 30).astype(dtype)
        s = dtype(scale)
        out = rows @ kernel + params["proj"]["bias"].reshape(30).astype(dtype) + s * ((rows @ down) @ up)
        return (out.reshape(16, 5, 6), (s * (rows.T @ (seed @ up.T))).reshape(6, 8, 4),
                (s * ((rows @ down).T @ seed)).reshape(4, 5, 6))

    assert tree["params"]["proj"]["kernel"].shape == (6, 8, 5, 6)
    references, truths = oracle(np.float32), oracle(np.float64)
    for label, dew, reference, truth in zip(
            ("the output", "lora_A's gradient", "lora_B's gradient"),
            (forward, gradient["lora_A"], gradient["lora_B"]), references, truths, strict=True):
        assert_as_exact_as_the_reference(np.asarray(dew), reference, truth, label)
