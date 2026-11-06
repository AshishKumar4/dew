"""The fusion search (docs/design/algebra.md §2 "Fusion", `dew.algebra.fusion`): a program's products and
their elementwise consumers fused into the GEMM template's epilogue by the compile-time pass, where the
template takes the product and its price says fused wins. Every program computes what it does as written,
its gradients too, whatever the pass fuses."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew import algebra
from dew.nn.kernels import generation


def draw(seed: int = 7):
    """Integer operands whose sums pass bf16's 256, so a dot's own rounding to bf16 matters, every product
    and fp32 sum exact."""
    keys = jax.random.split(jax.random.key(seed), 4)
    return (jax.random.randint(keys[0], (256, 512), -7, 8).astype(jnp.bfloat16),
            jax.random.randint(keys[1], (512, 256), -7, 8).astype(jnp.bfloat16),
            jax.random.randint(keys[2], (256,), -3, 4).astype(jnp.bfloat16),
            jax.random.randint(keys[3], (256, 256), -3, 4).astype(jnp.bfloat16))


def biased(x, w, bias, other):
    """A projection, its bias and a scale after it."""
    del other
    return (x @ w + bias) * jnp.bfloat16(2)


def escaping(x, w, bias, other):
    """A projection whose product is read after its consumers too."""
    del other
    product = x @ w
    return product + bias, product


def widened(x, w, bias, other):
    """A projection rounded to bf16, then widened and added to in float32."""
    del other
    return (x @ w).astype(jnp.float32) + bias.astype(jnp.float32)


def chained(x, w, bias, other):
    """A projection whose consumer feeds a second projection, its own consumer after it."""
    hidden = jnp.maximum(x @ w + bias, jnp.bfloat16(0))
    return hidden @ other - bias


LAYERS = [biased, escaping, widened, chained]


def as_written(layer, *operands):
    return jax.jit(layer)(*operands)


def rewritten(layer, *operands):
    return jax.jit(algebra.rewritten(layer))(*operands)


def assert_equal(got, expected):
    for value, reference in zip(jax.tree.leaves(got), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(np.asarray(value, np.float32), np.asarray(reference, np.float32))


ARMS = [("sm90", {"grouped_matmul": "mosaic_gpu"}), ("sm80", {"grouped_matmul": "mosaic_gpu"}), ("sm90", {}),
        ("sm80", {})]
"""Each GPU and the kernels a call names: an H100's template named for its projections (an A/B's arm, where
the pass fuses their consumers into its epilogue), an A100's (whose template takes no side operand, so the
pass keeps the program as traced), and an H100's and an A100's selectors, which give a dense projection
XLA's dot."""


@pytest.mark.parametrize("layer", LAYERS, ids=lambda layer: layer.__name__)
@pytest.mark.parametrize(("generation_name", "named"), ARMS, ids=["sm90-named", "sm80-named", "sm90", "sm80"])
def test_a_layer_under_the_pass_computes_what_it_does_as_written(layer, generation_name, named):
    """Each layer under the pass computes what it does as written, bitwise, in each arm (`ARMS`): an
    epilogue reads its product as the dot rounds it, a product read after its consumers is still its own,
    a widened product widens the dot's rounding."""
    operands = draw()
    with generation.assumed(generation_name), generation.chosen(**named):
        assert_equal(rewritten(layer, *operands), as_written(layer, *operands))


@pytest.mark.parametrize("layer", [biased, chained], ids=lambda layer: layer.__name__)
def test_a_layers_gradient_under_the_pass_is_its_gradient_as_written(layer):
    """The gradient of each layer's sum against a fixed cotangent, under the pass on a GPU whose template a
    call names (its backward's products candidates of their own), is the gradient as written, bitwise."""
    operands = draw()
    cotangent = jax.random.randint(jax.random.key(11), (256, 256), -2, 3).astype(jnp.bfloat16)

    def loss(f):
        return lambda *xs: jnp.sum((f(*xs) * cotangent).astype(jnp.float32))

    grad = jax.grad(loss(layer), argnums=(0, 1, 2))
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic_gpu"):
        assert_equal(jax.jit(algebra.rewritten(grad))(*operands), jax.jit(grad)(*operands))
