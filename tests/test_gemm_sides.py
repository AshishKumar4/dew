"""The GEMM template's side operands by group (`axes.GROUPS`, kernel-templates.md Appendix C row 10): a
bias of each group's own on the epilogue, and a scale of each group's own on the prologue, read at each
tile's group in the template's streams, their outputs and gradients held to check 1."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import generations
from reference_error import assert_as_exact
from test_gemm_template import DEPTH, ROUTED, ROWS, SEEDS, SIZES, WIDTH, _draw, biased_row

from dew.nn.kernels import gemm, generation


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
def test_a_per_group_side_operand_and_its_gradient_hold_check_1(generation_name):
    """A bias of each group's own, `[G, N]` on the axes `('groups', 'columns')` (an expert's,
    kernel-templates.md §3, Appendix C row 10): the template's output and its gradient through the bias, a
    sum over each group's rows, on par with the program that adds each row its group's bias over the draws
    (check 1), truth in float64, where the side operand's groups axis was refused (row 3642)."""
    sizes = jnp.asarray(SIZES)

    def draw(seed):
        x, w, *_ = _draw(seed)
        rng = np.random.default_rng(seed + 3000)
        return x, w, jnp.asarray(rng.normal(size=(len(SIZES), WIDTH)), jnp.float32)

    def dew(x, w, b):
        def loss(b):
            # The groups axis reaches the hook as rows: each row's tile of its group's bias.
            out = gemm.gemm(x, w, sizes, epilogue=biased_row,
                                epilogue_operands=((b, ("groups", "columns")),), out_dtype=jnp.float32,
                                interpret_on_cpu=True)
            return jnp.sum(out[:ROUTED] ** 2), out
        (_, out), grad = jax.value_and_grad(loss, has_aux=True)(b)
        return out[:ROUTED], grad

    def plain(x, w, b, wide=jnp.float32):
        def loss(b):
            # The matrix in the rows' dtype, as the template multiplies it.
            matrix = w.astype(x.dtype).astype(wide)
            product = jax.lax.ragged_dot(x.astype(wide), matrix, sizes, preferred_element_type=wide)
            out = product + jnp.repeat(b.astype(wide), sizes, axis=0, total_repeat_length=ROWS)
            return jnp.sum(out[:ROUTED] ** 2), out
        (_, out), grad = jax.value_and_grad(loss, has_aux=True)(b)
        return out[:ROUTED], grad

    with generation.assumed(generation_name):
        assert_as_exact(jax.jit(dew), jax.jit(plain), draw, "a per-group bias", seeds=SEEDS,
                        truth_fn=functools.partial(plain, wide=jnp.float64))


@pytest.mark.parametrize("generation_name", generations("sm90", "v5e"))
def test_a_per_group_prologue_side_operand_and_its_gradients_hold_check_1(generation_name):
    """A prologue scaling each row's depth by its group's own vector, `[G, K]` on the axes
    `('groups', 'depth')` (an expert's input scale): the template's launch named (`'mosaic'`, a GPU's
    warpgroup MMA or a TPU core's MXU), reading the side at each tile's group in its streams
    (`mosaic_gpu._gmm_views`, the kernel gradient's tile at its group, `tgmm`), the output and every gradient
    (rows, matrix and scales) on par with the program that scales each row by its group's vector over the
    draws (check 1), truth in float64; where the template refused a prologue side by group (Triton's or the
    TPU's old launch read it)."""
    sizes = jnp.asarray(SIZES)

    def draw(seed):
        x, w, *_ = _draw(seed)
        rng = np.random.default_rng(seed + 4000)
        return x, w.astype(x.dtype), jnp.asarray(rng.normal(size=(len(SIZES), DEPTH)), jnp.float32)

    def scaled(x, s):
        return x * s

    # A bf16 output, so every product of the reverse pass multiplies 16-bit operands, as the launch takes.
    def dew(x, w, s):
        def loss(x, w, s):
            out = gemm.gemm(x, w, sizes, prologue=scaled, prologue_operands=((s, ("groups", "depth")),),
                            out_dtype=jnp.bfloat16, interpret_on_cpu=True)
            return jnp.sum(out[:ROUTED].astype(jnp.float32) ** 2), out
        (_, out), grads = jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True)(x, w, s)
        return out[:ROUTED], *grads

    def plain(x, w, s, wide=jnp.float32):
        def loss(x, w, s):
            rows = jnp.repeat(s.astype(wide), sizes, axis=0, total_repeat_length=ROWS)
            left = (x.astype(wide) * rows).astype(x.dtype).astype(wide)
            out = jax.lax.ragged_dot(left, w.astype(wide), sizes, preferred_element_type=wide)
            out = out.astype(jnp.bfloat16).astype(wide) if wide == jnp.float32 else out
            return jnp.sum(out[:ROUTED] ** 2), out
        (_, out), grads = jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True)(x, w, s)
        return out[:ROUTED], *grads

    with (generation.assumed(generation_name), generation.chosen(grouped_matmul="mosaic"),
          jax.default_matmul_precision("default"), generation.routes() as routes):
        assert_as_exact(jax.jit(dew), jax.jit(plain), draw, "a per-group prologue side", seeds=SEEDS,
                        truth_fn=functools.partial(plain, wide=jnp.float64))
    assert {route.kernel for route in routes if route.choice == "grouped_matmul"} == {"mosaic"}


@pytest.mark.parametrize("generation_name", generations("sm90", "v5e"))
def test_a_side_operand_of_a_dimension_of_one_runs_on_the_template(generation_name):
    """Side operands with a dimension of one (a `[rows, 1]` mean square on the prologue, a `[1, width]` bias
    on the epilogue), squeezed and given back broadcast across the hook's tile (`hooks._unsqueezed`):
    the template's launch named, its output and every gradient on par with the plain program over the draws
    (check 1), truth in float64, where Mosaic's layouts took no dimension of one and Triton's ran them."""
    sizes = jnp.asarray(SIZES)

    def draw(seed):
        x, w, ms, *_ = _draw(seed)
        rng = np.random.default_rng(seed + 5000)
        return (x, w.astype(x.dtype), ms, jnp.asarray(rng.normal(size=(1, WIDTH)), jnp.float32))

    def normed(x, ms):
        return (x.astype(jnp.float32) * jax.lax.rsqrt(ms)).astype(x.dtype)

    def biased(acc, b):
        return acc + b

    def dew(x, w, ms, b):
        def loss(x, w, ms, b):
            out = gemm.gemm(x, w, sizes, prologue=normed, prologue_operands=((ms[:, None], ("rows", None)),),
                            epilogue=biased, epilogue_operands=((b, (None, "columns")),),
                            out_dtype=jnp.bfloat16, interpret_on_cpu=True)
            return jnp.sum(out[:ROUTED].astype(jnp.float32) ** 2), out
        (_, out), grads = jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True)(x, w, ms, b)
        return out[:ROUTED], *grads

    def plain(x, w, ms, b, wide=jnp.float32):
        def loss(x, w, ms, b):
            left = (x.astype(wide) * jax.lax.rsqrt(ms.astype(wide))[:, None]).astype(x.dtype).astype(wide)
            out = jax.lax.ragged_dot(left, w.astype(wide), sizes, preferred_element_type=wide)
            out = out + b.astype(wide)
            out = out.astype(jnp.bfloat16).astype(wide) if wide == jnp.float32 else out
            return jnp.sum(out[:ROUTED] ** 2), out
        (_, out), grads = jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True)(x, w, ms, b)
        return out[:ROUTED], *grads

    with (generation.assumed(generation_name), generation.chosen(grouped_matmul="mosaic"),
          jax.default_matmul_precision("default"), generation.routes() as routes):
        assert_as_exact(jax.jit(dew), jax.jit(plain), draw, "sides of a dimension of one", seeds=SEEDS,
                        truth_fn=functools.partial(plain, wide=jnp.float64))
    assert {route.kernel for route in routes if route.choice == "grouped_matmul"} == {"mosaic"}
