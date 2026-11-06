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


def bare(x, w, bias, other):
    """A projection alone, nothing after it."""
    del bias, other
    return x @ w


def kernel_gradient(x, w, bias, other):
    """A kernel gradient's product, xᵀ @ dy: the rows contracted."""
    del w, bias
    return x.T @ other


SIZES = (96, 160)
"""Two groups' rows of `draw`'s 256."""


def grouped(x, w, bias, other):
    """Each group's rows times its matrix (`lax.ragged_dot`, an MoE's experts), a bias after it."""
    del other
    return jax.lax.ragged_dot(x, jnp.stack([w, -w]), jnp.asarray(SIZES, jnp.int32)) + bias


def grouped_kernel_gradient(x, w, bias, other):
    """Each group's rows of x contracted with its rows of dy, `[groups, depth, columns]`: the grouped
    kernel gradient's form (`lax.ragged_dot_general`, the rows ragged within the contraction)."""
    del w, bias
    numbers = jax.lax.RaggedDotDimensionNumbers(dot_dimension_numbers=(((0,), (0,)), ((), ())),
                                               lhs_ragged_dimensions=[0], rhs_group_dimensions=[])
    return jax.lax.ragged_dot_general(x, other, jnp.asarray(SIZES, jnp.int32), numbers)


LAYERS = [biased, escaping, widened, chained, bare, kernel_gradient, grouped, grouped_kernel_gradient]


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


@pytest.mark.parametrize("layer", [bare, kernel_gradient, grouped, grouped_kernel_gradient],
                         ids=lambda layer: layer.__name__)
def test_a_plain_programs_product_runs_the_template_its_call_names(layer):
    """A plain JAX program's product of each form the template takes (`fusion.Form`: alone, a kernel
    gradient's, each group's rows times its matrix, a grouped kernel gradient's), under the pass on an H100
    whose template a call names, runs the template's launch (its route reported as the selector's), where
    the program as written runs XLA's dot."""
    operands = draw()
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic_gpu"):
        with generation.routes() as written:
            jax.jit(layer).trace(*operands)
        with generation.routes() as passed:
            jax.jit(algebra.rewritten(layer)).trace(*operands)
    assert not any(route.choice == "grouped_matmul" for route in written)
    assert any(route.choice == "grouped_matmul" and route.kernel == "mosaic_gpu" for route in passed)


def _template_routes(trace) -> list:
    """The grouped matmul's routes `trace()` reports that run the template."""
    with generation.routes() as found:
        trace()
    return [route for route in found if route.choice == "grouped_matmul" and route.kernel == "mosaic_gpu"]


def test_a_plain_programs_gradient_runs_the_template_in_either_order_through_dew_jit():
    """`dew.jit` over a plain JAX loss on an H100 whose template a call names: its gradient's products run
    the template whether `jax.grad` is taken of the compiled program (inside a plain jit, the pass reads the
    program as it traces) or the program compiles the gradient (the pass reads the joint program), each
    the gradient as written, bitwise. The loss contracts its projection whole with a fixed cotangent, so
    the projection is a product alone: a fused epilogue's derivative recomputes its sums in IEEE fp32,
    which the template does not take, and a call naming it refuses."""
    import dew

    operands = draw()
    cotangent = jax.random.randint(jax.random.key(11), (256, 256), -2, 3).astype(jnp.bfloat16)

    def loss(x, w, bias, other):
        del bias, other
        return jnp.tensordot(x @ w, cotangent, axes=2).astype(jnp.float32)

    plain = jax.grad(loss, argnums=(0, 1))
    outside = jax.jit(jax.grad(dew.jit(loss), argnums=(0, 1)))
    inside = dew.jit(jax.grad(loss, argnums=(0, 1)))
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic_gpu"):
        assert not _template_routes(lambda: jax.jit(plain).trace(*operands))
        assert _template_routes(lambda: outside.trace(*operands))
        assert _template_routes(lambda: inside.trace(*operands))
        expected = jax.jit(plain)(*operands)
        assert_equal(outside(*operands), expected)
        assert_equal(inside(*operands), expected)
