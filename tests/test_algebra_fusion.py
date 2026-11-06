"""The fusion search (docs/design/algebra.md §2 "Fusion", `dew.algebra.fusion`): a program's products and
their elementwise consumers fused into the GEMM template's epilogue by the compile-time pass."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew import algebra
from dew.nn.kernels import fused_gemm, generation


def draw():
    """Integer operands whose sums pass bf16's 256, so the dot's own rounding to bf16 matters, every product
    and fp32 sum exact."""
    keys = jax.random.split(jax.random.key(7), 3)
    return (jax.random.randint(keys[0], (128, 128), -7, 8).astype(jnp.bfloat16),
            jax.random.randint(keys[1], (128, 256), -7, 8).astype(jnp.bfloat16),
            jax.random.randint(keys[2], (256,), -3, 4).astype(jnp.bfloat16))


def layer(x, w, bias):
    """A projection, its bias and a scale after it, as a model writes them."""
    return (x @ w + bias) * jnp.bfloat16(2)


@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
def test_a_products_consumers_are_fused_into_the_templates_epilogue(generation_name):
    """Under the pass, a projection's bias and scale run in the GEMM template's epilogue where the template
    takes them (the epilogue's route the fused kernel's), and the layer computes what it does as written,
    bitwise: the epilogue reads the product as the dot rounds it."""
    x, w, bias = draw()
    with generation.assumed(generation_name), generation.routes() as found:
        got = jax.jit(algebra.rewritten(layer))(x, w, bias)
    assert any(route.choice == fused_gemm.EPILOGUE.choice and route.kernel == "fused"
               for route in found), found
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(layer(x, w, bias), np.float32))


def test_a_product_the_template_does_not_take_is_kept_as_traced():
    """Where the template takes no product (a CPU's own kernels), the layer's program is kept as traced: no
    epilogue is fused, and it computes what it does as written."""
    x, w, bias = draw()
    with generation.routes() as found:
        got = jax.jit(algebra.rewritten(layer))(x, w, bias)
    assert not any(route.choice == fused_gemm.EPILOGUE.choice for route in found), found
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(layer(x, w, bias), np.float32))
